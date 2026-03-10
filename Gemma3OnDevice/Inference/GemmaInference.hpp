// GemmaInference.hpp
// Core C++ inference engine — ExecuTorch runtime + CoreML backend delegate
//
// Model forward signature (StaticCache):
//   inputs:  input_ids [1,1] int64,
//            cache_position [1] int64
//   outputs: logits [1,1,vocab_size] float32
//   KV-cache is mutable module state in the .pte bundle; no explicit KV
//   tensors are passed in or out by the C++ layer.
//
// Prefill reuses the same exported decode-step graph by running it once per
// prompt token (or a single call with seq_len == prompt_len if the export
// supports a prefill-length input — the default export traces at seq_len=1
// for decode, so prefill is handled by iterating over prompt tokens).
//
// This class is NOT thread-safe — concurrency is managed at the ObjC layer.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gemma {

// ---------------------------------------------------------------------------
// Generation configuration
// ---------------------------------------------------------------------------

/// Controls the text-generation sampling strategy and length limits.
struct GenerationConfig {
    /// Hard limit on the number of new tokens produced.
    int   max_new_tokens     = 512;

    /// Softmax temperature.  0.0 → greedy argmax, 1.0 → unscaled distribution.
    float temperature        = 0.7f;

    /// Nucleus (top-p) sampling threshold [0, 1].  1.0 disables nucleus sampling.
    float top_p              = 0.9f;

    /// Top-k sampling limit.  0 disables top-k filtering.
    int   top_k              = 40;

    /// Maximum total tokens (prompt + generation) the model will see.
    int   max_sequence_length = 2048;

    /// EOS token ID — generation halts when this token is sampled.
    int   eos_token_id       = 1;   // Gemma-3 <eos>; also stops on <end_of_turn>=107
};

// ---------------------------------------------------------------------------
// Streaming callback
// ---------------------------------------------------------------------------

/// Called on the inference thread for each decoded token fragment.
/// @param token     UTF-8 piece for this step (may be empty for leading spaces)
/// @param is_done   True on the final call (EOS or max_new_tokens reached)
using TokenCallback = std::function<void(const std::string& token, bool is_done)>;

// ---------------------------------------------------------------------------
// GemmaInference
// ---------------------------------------------------------------------------

class GemmaInference {
public:
     GemmaInference();
    ~GemmaInference();

    // Non-copyable — owns heap resources
    GemmaInference(const GemmaInference&)            = delete;
    GemmaInference& operator=(const GemmaInference&) = delete;

    /// Load model artefacts from the app bundle.
    ///
    /// Internally creates a torch::executor::util::FileDataLoader for the
    /// .pte file and initialises the ExecuTorch Method.
    ///
    /// @param model_path      Absolute path to gemma3_4b_int4_coreml.pte
    /// @param tokenizer_path  Absolute path to tokenizer.model (SentencePiece)
    /// @return true on success; logs an ET_LOG(Error, …) on failure
    bool load(const std::string& model_path,
              const std::string& tokenizer_path);

    /// Run autoregressive text generation.
    ///
    /// Formats the prompt using the Gemma-3 instruct template, runs prefill,
    /// then calls the decode loop until EOS or max_new_tokens is reached.
    /// Each decoded token fragment is delivered via @p callback on the calling
    /// thread (synchronous — the caller should dispatch onto a background
    /// thread if needed).
    ///
    /// @param prompt   Raw user text (will be wrapped in Gemma chat template)
    /// @param config   Sampling and length configuration
    /// @param callback Streaming token callback
    void generate(const std::string&   prompt,
                  const GenerationConfig& config,
                  TokenCallback           callback);

    /// Returns true once load() has succeeded.
    bool isLoaded() const noexcept;

    /// Human-readable summary of the loaded model (vocab size, seq len, …).
    std::string modelInfo() const;

    /// Signal the running generate() call to stop at the next token boundary.
    /// Safe to call from any thread.  Automatically cleared at the start of
    /// the next generate() call.
    void requestCancel() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // ---- helpers ----

    /// Reset the position pointer to 0 so the next generate() call
    /// overwrites cache entries from the beginning.
    void resetKVCache();

    /// Run one forward step through the ExecuTorch method.
    ///
    /// Sets input_ids and cache_position, executes "forward", and copies
    /// the output logits into out_logits.  KV-cache state is updated
    /// in-place inside the .pte method (StaticCache); no explicit KV
    /// tensors are passed or returned.
    ///
    /// @param token_id      The single token to process this step.
    /// @param cache_pos     Absolute position in the KV-cache for this step.
    /// @param out_logits    Pointer to the impl_->logit_scratch buffer.
    /// @return Error::Ok on success.
    torch::executor::Error runStep(int32_t token_id,
                                   int32_t cache_pos,
                                   float*  out_logits);

    /// Sample one token ID from a raw logit vector.
    int32_t sampleToken(const float* logits, int vocab_size,
                        const GenerationConfig& config);

    /// Apply temperature scaling, top-k filter, and top-p nucleus filter,
    /// then sample from the resulting distribution.
    int32_t sampleFiltered(std::vector<std::pair<float,int>>& candidates,
                           float temperature, float top_p);
};

} // namespace gemma
