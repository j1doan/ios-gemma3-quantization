// Tokenizer.cpp
// SentencePiece tokenizer implementation for Gemma-3

#include "Tokenizer.hpp"

// SentencePiece public C++ header — shipped with the sentencepiece XCFramework
#include <sentencepiece_processor.h>

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace gemma {

// ---------------------------------------------------------------------------
GemmaTokenizer::GemmaTokenizer() = default;
GemmaTokenizer::~GemmaTokenizer() = default;

// ---------------------------------------------------------------------------
bool GemmaTokenizer::load(const std::string& path) {
    sp_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    const auto status = sp_->Load(path);
    if (!status.ok()) {
        sp_.reset();
        return false;
    }
    loaded_ = true;
    return true;
}

// ---------------------------------------------------------------------------
std::vector<int32_t> GemmaTokenizer::encode(const std::string& text,
                                             bool add_bos,
                                             bool add_eos) const {
    assert(loaded_ && "Tokenizer not loaded");
    std::vector<int> ids;
    sp_->Encode(text, &ids);

    std::vector<int32_t> result;
    result.reserve(ids.size() + (add_bos ? 1 : 0) + (add_eos ? 1 : 0));
    if (add_bos)  result.push_back(SpecialTokens::kBos);
    for (int id : ids) result.push_back(static_cast<int32_t>(id));
    if (add_eos)  result.push_back(SpecialTokens::kEos);
    return result;
}

// ---------------------------------------------------------------------------
std::vector<int32_t> GemmaTokenizer::encodePrompt(const std::string& user_text) const {
    assert(loaded_ && "Tokenizer not loaded");

    // Gemma-3 instruct template (identical to Gemma 2 template):
    //   <bos><start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n
    std::ostringstream oss;
    oss << "<start_of_turn>user\n"
        << user_text
        << "<end_of_turn>\n"
        << "<start_of_turn>model\n";

    // BOS + encoded template text (the control tokens are in the SP vocab as
    // full literal strings, so a plain Encode() handles them correctly)
    return encode(oss.str(), /*add_bos=*/true, /*add_eos=*/false);
}

// ---------------------------------------------------------------------------
std::string GemmaTokenizer::decode(const std::vector<int32_t>& ids) const {
    assert(loaded_ && "Tokenizer not loaded");
    std::vector<int> int_ids(ids.begin(), ids.end());
    std::string result;
    sp_->Decode(int_ids, &result);
    return result;
}

// ---------------------------------------------------------------------------
std::string GemmaTokenizer::decodeToken(int32_t id) const {
    assert(loaded_ && "Tokenizer not loaded");
    return sp_->IdToPiece(id);
}

// ---------------------------------------------------------------------------
int GemmaTokenizer::vocabSize() const {
    if (!loaded_) return 0;
    return sp_->GetPieceSize();
}

} // namespace gemma
