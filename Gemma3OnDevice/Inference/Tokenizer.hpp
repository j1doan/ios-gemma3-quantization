// Tokenizer.hpp
// SentencePiece-based tokenizer wrapper for Gemma-3
//
// Gemma-3 uses a 256 000-token SentencePiece vocabulary identical to
// the one shipped with gemma_pytorch.  This header exposes the minimal
// interface needed by GemmaInference (encode / decode).
//
// Dependencies: link against libsentencepiece (available via CocoaPods or
// the sentencepiece XCFramework bundled with ExecuTorch iOS artefacts).

#pragma once

#include <string>
#include <vector>

// Forward-declare the SentencePiece processor to avoid pulling the full
// header into every translation unit.
namespace sentencepiece { class SentencePieceProcessor; }

namespace gemma {

/// Gemma-3 special token IDs (SentencePiece vocab)
struct SpecialTokens {
    static constexpr int kPad   = 0;
    static constexpr int kEos   = 1;
    static constexpr int kBos   = 2;
    static constexpr int kUnk   = 3;

    // Gemma-3 multi-turn control tokens (added on top of base SP vocab)
    // These are represented as raw bytes encoded as a single SP "user token"
    // <start_of_turn>  →  token id 106  (0x6A  in the Gemma-3 SP model)
    // <end_of_turn>    →  token id 107
    static constexpr int kStartOfTurn = 106;
    static constexpr int kEndOfTurn   = 107;
};

/// SentencePiece tokenizer for Gemma-3.
///
/// Thread-compatible but NOT thread-safe — create one instance per thread,
/// or guard concurrent calls with a mutex.
class GemmaTokenizer {
public:
    GemmaTokenizer();
    ~GemmaTokenizer();

    // Non-copyable (underlying processor holds file resources)
    GemmaTokenizer(const GemmaTokenizer&)            = delete;
    GemmaTokenizer& operator=(const GemmaTokenizer&) = delete;

    /// Load the SentencePiece model from disk.
    /// @param path Absolute path to tokenizer.model
    /// @return true on success
    bool load(const std::string& path);

    /// Encode a plain UTF-8 string into token IDs.
    /// @param text     Input text
    /// @param add_bos  Prepend the BOS (Beginning-Of-Sequence) token
    /// @param add_eos  Append the EOS token
    std::vector<int32_t> encode(const std::string& text,
                                bool add_bos = false,
                                bool add_eos = false) const;

    /// Encode a full Gemma-3 chat prompt following the instruct template:
    ///   <bos><start_of_turn>user\n{user_text}<end_of_turn>\n
    ///        <start_of_turn>model\n
    std::vector<int32_t> encodePrompt(const std::string& user_text) const;

    /// Decode a sequence of token IDs back to UTF-8 text.
    std::string decode(const std::vector<int32_t>& ids) const;

    /// Decode a single token ID.
    std::string decodeToken(int32_t id) const;

    /// Vocabulary size reported by the loaded SP model.
    int vocabSize() const;

    bool isLoaded() const noexcept { return loaded_; }

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_;
    bool loaded_ = false;
};

} // namespace gemma
