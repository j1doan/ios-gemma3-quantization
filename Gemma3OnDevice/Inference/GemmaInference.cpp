// GemmaInference.cpp
// ExecuTorch + CoreML inference implementation for Gemma-3
//
// Key change vs. the original: the model is exported with a static KV-cache
// (key/value tensors of shape [1, num_kv_heads, MAX_CACHE_LEN, head_dim]).
// Each generate step processes exactly ONE token, so the per-step cost is
// O(N * d) instead of the O(N² * d) full-prefix re-computation of the
// original stateless version.
//
// Build requirements:
//   - Link: executorch.xcframework (or libexecutorch.a + libexecutorch_core.a)
//   - Link: coreml_backend.xcframework  (registers CoreML delegate at startup)
//   - Link: xnnpack_backend.xcframework (fallback ops)
//   - Link: sentencepiece.xcframework
//   - Header search paths: executorch source root

#include "GemmaInference.hpp"
#include "Tokenizer.hpp"

// --- ExecuTorch runtime -------------------------------------------------
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>       // torch::executor::runtime_init()
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

// When the CoreML backend XCFramework is linked, its +load / __attribute__((constructor))
// registers the delegate automatically — no explicit include needed at runtime.

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace gemma {

using namespace torch::executor;

// Gemma-3 Instruct emits <end_of_turn> (token 107) before the bare EOS token.
// Generation must stop on either ID.
static constexpr int32_t kEndOfTurnTokenId = 107;

// Must match MAX_CACHE_LEN used when constructing StaticCache in export_gemma3.py.
static constexpr int32_t MAX_CACHE_LEN = 2048;

// ---------------------------------------------------------------------------
// Private implementation struct
// ---------------------------------------------------------------------------

struct GemmaInference::Impl {
    // ---- ExecuTorch objects ------------------------------------------------

    std::unique_ptr<util::FileDataLoader>        loader;
    std::unique_ptr<Program>                     program;
    // Loaded "forward" method.
    // With StaticCache the method signature is:
    //   inputs:  input_ids [1,1] int64,  cache_position [1] int64
    //   outputs: logits [1,1,vocab_size] float32
    // KV-cache is mutable module state inside the .pte bundle; no explicit
    // KV tensors are passed in or out by the C++ layer.
    std::unique_ptr<Method>                      method;
    std::unique_ptr<util::MallocMemoryAllocator> method_allocator;
    std::vector<std::vector<uint8_t>>            planned_buffers;

    // ---- Autoregressive position tracking ---------------------------------
    // Tracks the absolute token position for the next decode step.
    // Reset to 0 at the start of each generate() call.
    int32_t next_cache_pos = 0;

    // ---- Scratch buffer for logits (size = vocab_size) --------------------
    std::vector<float> logit_scratch;

    // ---- Model state -------------------------------------------------------
    std::unique_ptr<GemmaTokenizer> tokenizer;

    bool loaded     = false;
    int  vocab_size = 262144;  // Gemma-3 default; overridden after load

    // ---- Sampling ----------------------------------------------------------
    std::mt19937 rng{std::random_device{}()};

    // ---- Cancellation ------------------------------------------------------
    std::atomic<bool> cancel_requested{false};
};

// ---------------------------------------------------------------------------
GemmaInference::GemmaInference()  : impl_(std::make_unique<Impl>()) {}
GemmaInference::~GemmaInference() = default;

bool GemmaInference::isLoaded() const noexcept { return impl_->loaded; }

void GemmaInference::requestCancel() noexcept {
    impl_->cancel_requested.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

bool GemmaInference::load(const std::string& model_path,
                           const std::string& tokenizer_path) {
    // The ExecuTorch PAL must be initialised exactly once per process.
    // runtime_init() is idempotent after the first call.
    torch::executor::runtime_init();

    // 1. Tokenizer ----------------------------------------------------------
    impl_->tokenizer = std::make_unique<GemmaTokenizer>();
    if (!impl_->tokenizer->load(tokenizer_path)) {
        ET_LOG(Error, "GemmaInference: tokenizer load failed: %s",
               tokenizer_path.c_str());
        return false;
    }
    impl_->vocab_size = impl_->tokenizer->vocabSize();

    // 2. FileDataLoader -----------------------------------------------------
    //    FileDataLoader::from() opens the file and verifies its header.
    //    It uses mmap on Darwin (iOS) for zero-copy weight access.
    auto loader_result = util::FileDataLoader::from(
        model_path.c_str(),
        /*alignment=*/alignof(std::max_align_t));
    if (!loader_result.ok()) {
        ET_LOG(Error, "GemmaInference: FileDataLoader failed for: %s  (err %d)",
               model_path.c_str(), (int)loader_result.error());
        return false;
    }
    impl_->loader =
        std::make_unique<util::FileDataLoader>(std::move(loader_result.get()));

    // 3. Program ------------------------------------------------------------
    //    Minimal verification is sufficient — the export script already runs
    //    full verification offline.
    auto program_result = Program::load(
        impl_->loader.get(),
        Program::Verification::Minimal);
    if (!program_result.ok()) {
        ET_LOG(Error, "GemmaInference: Program::load failed (err %d)",
               (int)program_result.error());
        return false;
    }
    impl_->program =
        std::make_unique<Program>(std::move(program_result.get()));

    // 4. Method metadata & planned memory -----------------------------------
    auto meta_result = impl_->program->method_meta("forward");
    if (!meta_result.ok()) {
        ET_LOG(Error, "GemmaInference: method_meta('forward') failed");
        return false;
    }
    const MethodMeta& meta = meta_result.get();

    impl_->planned_buffers.clear();
    std::vector<Span<uint8_t>> planned_spans;
    const size_t n_buffers = meta.num_memory_planned_buffers();
    for (size_t i = 0; i < n_buffers; ++i) {
        const size_t buf_sz =
            static_cast<size_t>(meta.memory_planned_buffer_size(i).get());
        impl_->planned_buffers.emplace_back(buf_sz, static_cast<uint8_t>(0));
        planned_spans.push_back(
            {impl_->planned_buffers.back().data(), buf_sz});
    }
    HierarchicalAllocator planned_memory(
        {planned_spans.data(), planned_spans.size()});

    // 5. Memory manager & method load ---------------------------------------
    impl_->method_allocator =
        std::make_unique<util::MallocMemoryAllocator>();
    MemoryManager memory_manager(impl_->method_allocator.get(),
                                 &planned_memory);

    auto method_result =
        impl_->program->load_method("forward", &memory_manager);
    if (!method_result.ok()) {
        ET_LOG(Error, "GemmaInference: load_method('forward') failed (err %d)",
               (int)method_result.error());
        return false;
    }
    impl_->method =
        std::make_unique<Method>(std::move(method_result.get()));

    // 6. Verify the method has the expected 2-input (StaticCache) signature --
    //    input 0: input_ids      [1, 1]  int64
    //    input 1: cache_position [1]     int64
    //    KV-cache is mutable module state inside the .pte bundle.
    const size_t num_inputs = meta.num_inputs();
    if (num_inputs < 2) {
        ET_LOG(Error, "GemmaInference: unexpected number of method inputs: %zu",
               num_inputs);
        return false;
    }

    // 7. Allocate logit scratch buffer --------------------------------------
    impl_->logit_scratch.resize(static_cast<size_t>(impl_->vocab_size));
    impl_->next_cache_pos = 0;

    impl_->loaded = true;
    ET_LOG(Info, "GemmaInference: loaded (vocab=%d)", impl_->vocab_size);
    return true;
}

// ---------------------------------------------------------------------------
// resetKVCache()
// ---------------------------------------------------------------------------

void GemmaInference::resetKVCache() {
    // KV-cache buffers live inside the .pte method state and are updated
    // in-place during execute().  Resetting the position pointer is sufficient:
    // the model's causal mask ensures positions beyond next_cache_pos are
    // never attended to, so stale entries from a prior turn are harmless.
    impl_->next_cache_pos = 0;
}

// ---------------------------------------------------------------------------
// runStep()  — single forward step with KV-cache
// ---------------------------------------------------------------------------

Error GemmaInference::runStep(int32_t token_id,
                               int32_t cache_pos,
                               float*  out_logits) {
    // input_ids [1, 1]
    int64_t ids_val = static_cast<int64_t>(token_id);
    TensorImpl::SizesType   ids_sizes[2]   = {1, 1};
    TensorImpl::StridesType ids_strides[2] = {1, 1};
    TensorImpl ids_impl(ScalarType::Long, 2, ids_sizes, &ids_val, ids_strides);
    Tensor input_ids(&ids_impl);

    // cache_position [1]
    int64_t pos_val = static_cast<int64_t>(cache_pos);
    TensorImpl::SizesType   pos_sizes[1]   = {1};
    TensorImpl::StridesType pos_strides[1] = {1};
    TensorImpl pos_impl(ScalarType::Long, 1, pos_sizes, &pos_val, pos_strides);
    Tensor cache_position(&pos_impl);

    // ids_impl and pos_impl are in scope for the entire function; both
    // TensorImpl pointers remain valid through execute().
    if (auto err = impl_->method->set_input(EValue(input_ids),      0);
            err != Error::Ok) return err;
    if (auto err = impl_->method->set_input(EValue(cache_position), 1);
            err != Error::Ok) return err;

    if (auto err = impl_->method->execute(); err != Error::Ok) return err;

    // Output 0: logits [1, 1, vocab_size] — KV-cache is updated in-place
    // inside the .pte module state; no explicit KV outputs to copy.
    const EValue& out0       = impl_->method->get_output(0);
    const float*  logits_ptr = out0.toTensor().const_data_ptr<float>();
    std::copy(logits_ptr,
              logits_ptr + impl_->vocab_size,
              out_logits);

    return Error::Ok;
}

// ---------------------------------------------------------------------------
// generate()
// ---------------------------------------------------------------------------

void GemmaInference::generate(const std::string&      prompt,
                               const GenerationConfig& config,
                               TokenCallback           callback) {
    if (!impl_->loaded) {
        ET_LOG(Error, "GemmaInference::generate called before load()");
        if (callback) callback("", true);
        return;
    }

    // Reset the KV-cache so each conversation turn starts from a clean state.
    resetKVCache();
    // Clear any pending cancellation from a previous call.
    impl_->cancel_requested.store(false, std::memory_order_relaxed);

    // 1. Tokenise with Gemma-3 instruct template
    const std::vector<int32_t> prompt_tokens =
        impl_->tokenizer->encodePrompt(prompt);

    const int max_seq = config.max_sequence_length;

    // Clamp the cache so we never overflow MAX_CACHE_LEN
    const int effective_max_seq = std::min(max_seq, MAX_CACHE_LEN);

    // 2. Prefill: feed each prompt token through the model to populate the
    //    KV-cache.  We discard the intermediate logits and only use the
    //    logits produced by the LAST prefill token to kick off decoding.

    float* logit_buf = impl_->logit_scratch.data();

    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        if (impl_->next_cache_pos >= effective_max_seq) {
            // Prompt alone fills the entire cache — abort gracefully.
            ET_LOG(Error, "generate: prompt length exceeds effective_max_seq (%d)",
                   effective_max_seq);
            if (callback) callback("", true);
            return;
        }
        const Error err = runStep(prompt_tokens[i],
                                  impl_->next_cache_pos,
                                  logit_buf);
        if (err != Error::Ok) {
            ET_LOG(Error, "generate: prefill runStep failed at pos %d (err %d)",
                   impl_->next_cache_pos, (int)err);
            if (callback) callback("", true);
            return;
        }
        ++impl_->next_cache_pos;
    }

    // 3. Autoregressive decode loop
    int n_new = 0;
    // Sample first token from last prefill logits
    int32_t next_id = sampleToken(logit_buf, impl_->vocab_size, config);

    while (impl_->next_cache_pos < effective_max_seq) {

        // Honour a cancellation request — stop at the next token boundary
        // so the ANE is not left mid-computation.
        if (impl_->cancel_requested.load(std::memory_order_relaxed)) {
            break;
        }

        // Stop on bare EOS or Gemma-3 Instruct's <end_of_turn> (token 107).
        // The instruct model always emits <end_of_turn> before bare EOS, so
        // checking only eos_token_id would cause the model to run past the
        // turn boundary and output the literal <end_of_turn> string.
        if (next_id == config.eos_token_id || next_id == kEndOfTurnTokenId) {
            if (callback) callback("", /*is_done=*/true);
            return;
        }

        // Decode and stream the token
        std::string piece = impl_->tokenizer->decodeToken(next_id);
        // Translate SentencePiece space marker (▁ = U+2581, UTF-8: E2 96 81)
        if (piece.size() >= 3 &&
            static_cast<unsigned char>(piece[0]) == 0xE2 &&
            static_cast<unsigned char>(piece[1]) == 0x96 &&
            static_cast<unsigned char>(piece[2]) == 0x81) {
            piece = " " + piece.substr(3);
        }
        if (callback) callback(piece, /*is_done=*/false);
        ++n_new;

        // Skip the next decode step if we've hit the token limit — avoids a
        // wasted runStep + sampleToken whose results would be discarded.
        if (n_new >= config.max_new_tokens) break;

        // Run one decode step to get logits for the NEXT token
        const Error err = runStep(next_id,
                                  impl_->next_cache_pos,
                                  logit_buf);
        if (err != Error::Ok) {
            ET_LOG(Error, "generate: decode runStep failed at pos %d (err %d)",
                   impl_->next_cache_pos, (int)err);
            break;
        }
        ++impl_->next_cache_pos;

        next_id = sampleToken(logit_buf, impl_->vocab_size, config);
    }

    // Reached max_new_tokens or max_sequence_length
    if (callback) callback("", /*is_done=*/true);
}

// ---------------------------------------------------------------------------
// sampleToken() — dispatch to filtered sampling
// ---------------------------------------------------------------------------

int32_t GemmaInference::sampleToken(const float*           logits,
                                     int                    vocab_size,
                                     const GenerationConfig& config) {
    // Greedy path for temperature == 0
    if (config.temperature <= 0.0f) {
        return static_cast<int32_t>(
            std::max_element(logits, logits + vocab_size) - logits);
    }

    // Build (logit, id) pairs and apply top-k truncation up front
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(static_cast<size_t>(vocab_size));
    for (int i = 0; i < vocab_size; ++i) {
        candidates.emplace_back(logits[i], i);
    }

    // Top-k filter
    const int k = (config.top_k > 0 && config.top_k < vocab_size)
                      ? config.top_k : vocab_size;
    std::partial_sort(candidates.begin(),
                      candidates.begin() + k,
                      candidates.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;  // descending
                      });
    candidates.resize(static_cast<size_t>(k));

    return sampleFiltered(candidates, config.temperature, config.top_p);
}

// ---------------------------------------------------------------------------
// sampleFiltered() — temperature + top-p + multinomial draw
// ---------------------------------------------------------------------------

int32_t GemmaInference::sampleFiltered(std::vector<std::pair<float, int>>& cands,
                                        float temperature,
                                        float top_p) {
    // Temperature scaling
    for (auto& [logit, _] : cands) logit /= temperature;

    // Softmax (numerically stable)
    const float max_logit = cands.front().first;
    float sum_exp = 0.0f;
    for (auto& [logit, _] : cands) {
        logit = std::exp(logit - max_logit);
        sum_exp += logit;
    }
    for (auto& [prob, _] : cands) prob /= sum_exp;

    // Top-p (nucleus) filtering — keep smallest prefix whose cumulative
    // probability exceeds top_p
    if (top_p < 1.0f) {
        float cumulative = 0.0f;
        size_t cutoff = cands.size();
        for (size_t i = 0; i < cands.size(); ++i) {
            cumulative += cands[i].first;
            if (cumulative >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        cands.resize(cutoff);
        // Re-normalise
        const float normaliser = cumulative;
        for (auto& [prob, _] : cands) prob /= normaliser;
    }

    // Multinomial draw
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(impl_->rng);
    for (const auto& [prob, id] : cands) {
        r -= prob;
        if (r <= 0.0f) return static_cast<int32_t>(id);
    }
    return static_cast<int32_t>(cands.back().second);
}

// ---------------------------------------------------------------------------
// modelInfo()
// ---------------------------------------------------------------------------

std::string GemmaInference::modelInfo() const {
    if (!impl_->loaded) return "Model not loaded";
    std::ostringstream oss;
    oss << "Gemma-3-4B-Instruct (INT4, CoreML, StaticCache)"
        << " | vocab=" << impl_->vocab_size;
    return oss.str();
}

} // namespace gemma
