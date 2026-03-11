// GemmaInference.hpp
// Core C++ inference engine — ExecuTorch runtime + CoreML backend delegate
//
// Model forward signature (StaticCache, text-only 2-input):
//   inputs:  input_ids [1,1] int64,
//            cache_position [1] int64
//   outputs: logits [1,1,vocab_size] float32
//
// Model forward signature (multimodal 4-input, exported with --multimodal):
//   inputs:  input_ids [1,1] int64,
//            cache_position [1] int64,
//            inputs_embeds [1,1,hidden_dim] float16,
//            embed_mask    [1,1,1] float16
//   outputs: logits [1,1,vocab_size] float32
//
// KV-cache is mutable module state in the .pte bundle; no explicit KV
// tensors are passed in or out by the C++ layer.
//
// When a vision encoder .pte is loaded (multimodal mode), images are:
//   1. Preprocessed to [1,3,896,896] on the Swift side
//   2. Passed through the vision encoder .pte → [1,256,3072] projected tokens
//   3. Injected into the decoder during prefill via embed_mask=1.0
//
// Prefill reuses the same exported decode-step graph by running it once per
// prompt token (or vision embedding during image prefill).
//
// This class is NOT thread-safe — concurrency is managed at the ObjC layer.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gemma {

// ---------------------------------------------------------------------------
// Architecture constants (Section 1 of edge spec)
// ---------------------------------------------------------------------------

static constexpr int kHiddenDim     = 3072;
static constexpr int kNImageTokens  = 256;   // projected vision tokens per image
static constexpr int kVisionInputH  = 896;
static constexpr int kVisionInputW  = 896;

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
    /// @param model_path           Absolute path to gemma3_4b_int4_coreml.pte
    /// @param tokenizer_path       Absolute path to tokenizer.model (SentencePiece)
    /// @param vision_encoder_path  Absolute path to gemma3_vision_encoder.pte
    ///                             (empty string to skip — text-only mode)
    /// @return true on success
    bool load(const std::string& model_path,
              const std::string& tokenizer_path,
              const std::string& vision_encoder_path = "");

    /// Run autoregressive text generation (text-only prompt).
    void generate(const std::string&      prompt,
                  const GenerationConfig& config,
                  TokenCallback           callback);

    /// Run autoregressive generation with an image.
    /// @param prompt         Raw user text
    /// @param pixel_data     Pre-processed pixel buffer [1,3,896,896] as float32
    ///                       (C-contiguous, CHW layout).  Must contain exactly
    ///                       3 * 896 * 896 floats.
    /// @param config         Sampling configuration
    /// @param callback       Streaming token callback
    void generateWithImage(const std::string&      prompt,
                           const float*            pixel_data,
                           const GenerationConfig& config,
                           TokenCallback           callback);

    bool isLoaded() const noexcept;

    /// True when a vision encoder .pte was loaded successfully.
    bool isMultimodal() const noexcept;

    std::string modelInfo() const;
    void requestCancel() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void resetKVCache();

    /// Run one text-decoder forward step.
    /// Adapts automatically to the 2-input (text-only) or 4-input (multimodal)
    /// model signature detected at load time.
    ///
    /// @param token_id       Token ID for this step (ignored when embed_data != nullptr)
    /// @param cache_pos      Absolute position in the KV-cache
    /// @param embed_data     If non-null, pointer to a [hidden_dim] float16 embedding
    ///                       to inject instead of looking up token_id
    /// @param out_logits     Output buffer (vocab_size floats)
    torch::executor::Error runStep(int32_t token_id,
                                   int32_t cache_pos,
                                   const void* embed_data,
                                   float*  out_logits);

    /// Run the vision encoder .pte on pre-processed pixel data.
    /// @param pixel_data  [1,3,896,896] float32 buffer  (3*896*896 floats)
    /// @param out_embeds  Output buffer: n_image_tokens * hidden_dim float16 values
    /// @return true on success
    bool runVisionEncoder(const float* pixel_data,
                          void*        out_embeds);

    int32_t sampleToken(const float* logits, int vocab_size,
                        const GenerationConfig& config);
    int32_t sampleFiltered(std::vector<std::pair<float,int>>& candidates,
                           float temperature, float top_p);

    /// Shared autoregressive decode loop used by both generate() and
    /// generateWithImage() after prefill is complete.
    void decodeLoop(float* logit_buf,
                    const GenerationConfig& config,
                    TokenCallback callback);

    /// Load the vision encoder .pte file.
    bool loadVisionEncoder(const std::string& path);
};

} // namespace gemma
