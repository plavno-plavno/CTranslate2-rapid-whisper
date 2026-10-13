#pragma once

#include <functional>
#include <optional>

#include "ctranslate2/decoding_utils.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/sampling.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {

  struct DecodingResult {
    std::vector<std::vector<size_t>> hypotheses;
    std::vector<float> scores;
    std::vector<std::vector<std::vector<float>>> attention;
    std::vector<std::vector<StorageView>> logits_vocab;
  };

  class SearchStrategy {
  public:
    virtual ~SearchStrategy() = default;
    virtual std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const std::vector<size_t>& end_ids,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool return_scores = false,
           const bool return_attention = false,
           const bool return_logits_vocab = true,
           const bool return_prefix = true,
           const size_t num_hypotheses = 1,
           const bool include_eos_in_hypotheses = true,
           const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors = {},
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const = 0;

    // Prefix tokens the caller already forwarded into the decoder state: they are not
    // in the hypotheses during the search but count in the length used to normalize scores.
    void set_hypothesis_length_offset(const dim_t offset) {
      _hypothesis_length_offset = offset;
    }

  protected:
    dim_t _hypothesis_length_offset = 0;
  };

  class BeamSearch : public SearchStrategy {
  public:
    BeamSearch(const dim_t beam_size,
               const float length_penalty = 0,
               const float coverage_penalty = 0,
               const float patience = 1);

    std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const std::vector<size_t>& end_ids,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool return_scores = false,
           const bool return_attention = false,
           const bool return_logits_vocab = true,
           const bool return_prefix = true,
           const size_t num_hypotheses = 1,
           const bool include_eos_in_hypotheses = true,
           const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors = {},
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;

  private:
    const dim_t _beam_size;
    const float _length_penalty;
    const float _coverage_penalty;
    const size_t _max_candidates;
  };

  class GreedySearch : public SearchStrategy {
  public:
    // Penalties are only applied to return scores consistent with the beam search.
    GreedySearch(const float length_penalty = 0,
                 const float coverage_penalty = 0);

    std::vector<DecodingResult>
    search(layers::Decoder& decoder,
           layers::DecoderState& state,
           const Sampler& sampler,
           const std::vector<size_t>& start_ids,
           const std::vector<size_t>& end_id,
           const dim_t start_step,
           const dim_t max_length,
           const dim_t min_length,
           const bool return_scores = false,
           const bool return_attention = false,
           const bool return_logits_vocab = true,
           const bool return_prefix = true,
           const size_t num_hypotheses = 1,
           const bool include_eos_in_hypotheses = true,
           const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors = {},
           const std::vector<std::vector<size_t>>* prefix_ids = nullptr) const override;

  private:
    const float _length_penalty;
    const float _coverage_penalty;
  };


  struct DecodingOptions {
    size_t beam_size = 1;
    float patience = 1;
    float length_penalty = 0;
    float coverage_penalty = 0;
    float repetition_penalty = 1;
    size_t no_repeat_ngram_size = 0;
    dim_t start_step = 0;
    size_t max_length = 256;
    size_t min_length = 0;
    size_t sampling_topk = 1;
    float sampling_topp = 1;
    float sampling_temperature = 1;
    size_t num_hypotheses = 1;
    bool include_eos_in_hypotheses = true;
    bool return_scores = false;
    bool return_attention = false;
    bool return_logits_vocab = false;
    bool return_prefix = true;
    std::vector<size_t> disable_ids;
    std::vector<size_t> disable_ids_begin;
    std::vector<std::shared_ptr<LogitsProcessor>> logits_processors;
    // Number of hard prefix tokens the caller forwarded into the decoder state in one
    // pass instead of passing them in start_tokens. Decoding then behaves as if they had
    // been forced step by step: they count in the hypothesis length for score
    // normalization and the first free step is handled as the step after a prefix.
    dim_t folded_prefix_length = 0;
  };

  std::vector<DecodingResult>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         std::vector<std::vector<size_t>> start_tokens,
         std::vector<size_t> end_ids,
         DecodingOptions options = DecodingOptions());

}
