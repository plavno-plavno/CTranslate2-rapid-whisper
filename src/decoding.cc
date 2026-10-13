#include "ctranslate2/decoding.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>

#include "ctranslate2/ops/ops.h"
#include "dispatch.h"

namespace ctranslate2 {

  static const ops::Gather gather;

  static void gather_beam_flat(StorageView& data, const StorageView& indices, dim_t beam_size) {
    merge_batch_beam(data);
    gather(data, indices);
    split_batch_beam(data, beam_size);
  }

  static void update_sample_with_prefix(const size_t step,
                                        StorageView& sampled_ids,
                                        StorageView& sampled_scores,
                                        const std::vector<std::vector<size_t>>& prefix_ids,
                                        const std::vector<size_t>& end_ids,
                                        const std::vector<dim_t>& batch_offset,
                                        const dim_t beam_size = 1,
                                        StorageView* beam_origins = nullptr,
                                        const bool is_expanded = true) {
    const dim_t batch_size = sampled_scores.dim(0);
    for (dim_t i = 0; i < batch_size; ++i) {
      const auto& prefix = prefix_ids[batch_offset[i]];
      if (step > prefix.size())
        continue;

      const dim_t num_samples = sampled_scores.dim(1);
      for (dim_t k = 0; k < num_samples; ++k) {
        const dim_t flat_index = i * num_samples + k;
        auto& sampled_id = sampled_ids.at<int32_t>(flat_index);
        int32_t new_id = -1;
        float new_score = 0;

        // When step < prefix_length, we override the sampled ids with the prefix ids
        // and set the highest probability to the first beam.
        if (step < prefix.size()) {
          new_id = prefix[step];
          new_score = (k == 0 ? 0.f : float(-1e10));

        // When step == prefix_length (the first unconstrained decoding step),
        // only the first beam is expanded. It happens that </s> appears in the topk,
        // especially when k is large. This can produce incorrect and short predictions
        // that dominate others when no length normalization is used (see issue #277).
        // To mitigate this issue, we penalize </s> in secondary beams.
        } else if (k > 0 && is_eos(sampled_id, end_ids)) {
          new_id = 0;
          new_score = -1e10;
        }

        if (new_id >= 0) {
          sampled_id = new_id;
          TYPE_DISPATCH(sampled_scores.dtype(), sampled_scores.at<T>(flat_index) = T(new_score));
          if (beam_origins)
            beam_origins->at<int32_t>(flat_index) = (is_expanded ? i * beam_size : i);
        }
      }
    }
  }

  static inline void convert_to_original_word_ids(const layers::Decoder& decoder,
                                                  StorageView& ids) {
    if (!decoder.output_layer_is_updated())
      return;
    ctranslate2::Device device = ids.device();
    if (device != Device::CPU)
      ids = ids.to(Device::CPU);
    auto* ids_data = ids.data<int32_t>();
    for (dim_t i = 0; i < ids.size(); ++i)
      ids_data[i] = decoder.to_original_word_id(ids_data[i]);
    if (ids.device() != device)
      ids = ids.to(device);
  }

  template <typename T>
  static void initialize_beam_scores(StorageView& scores,
                                     const dim_t batch_size,
                                     const dim_t beam_size) {
    const dim_t size = batch_size * beam_size;
    scores.resize({size});
    auto* data = scores.data<T>();
    for (dim_t i = 0; i < size; ++i) {
      data[i] = (i % beam_size == 0 ? T(0) : std::numeric_limits<T>::lowest());
    }
  }

  static StorageView unflatten_ids(StorageView& ids,
                                   const dim_t beam_size,
                                   const dim_t vocabulary_size,
                                   const bool is_expanded) {
    const dim_t num_ids = ids.size();
    StorageView beam_origins({num_ids}, DataType::INT32);

    auto* ids_data = ids.data<int32_t>();
    auto* origins_data = beam_origins.data<int32_t>();

    for (dim_t i = 0; i < num_ids; ++i) {
      const auto flat_id = ids_data[i];
      const auto beam_id = flat_id / vocabulary_size;
      const auto word_id = flat_id % vocabulary_size;
      const auto batch_id = i / ids.dim(-1);
      ids_data[i] = word_id;
      origins_data[i] = is_expanded ? batch_id * beam_size + beam_id : batch_id;
    }

    return beam_origins;
  }

  static void append_step_output(StorageView& history,    // [batch, beam, time, ...]
                                 StorageView step_output,  // [batch, beam, ...]
                                 const StorageView* beam_origins = nullptr) {
    step_output.expand_dims(2);  // Insert time dimension.

    if (history) {
      if (beam_origins)
        gather_beam_flat(history, *beam_origins, step_output.dim(1));
      const StorageView cur_history(std::move(history));
      ops::Concat(2)({&cur_history, &step_output}, history);
    } else {
      history = std::move(step_output);
    }
  }

  static std::vector<size_t> build_hypothesis(const StorageView& history,
                                              const dim_t batch,
                                              const dim_t beam,
                                              const dim_t start,
                                              const dim_t end) {
    const auto* ids = history.index<int32_t>({batch, beam, 0});
    return std::vector<size_t>(ids + start, ids + end);
  }

  static std::vector<std::vector<float>> build_attention(const StorageView& history,
                                                         const dim_t batch,
                                                         const dim_t beam,
                                                         const dim_t start,
                                                         const dim_t end) {
    if (!history)
      return {};

    const auto source_length = history.dim(-1);

    std::vector<std::vector<float>> attention;
    attention.reserve(end - start);
    for (dim_t t = start; t < end; ++t) {
      const auto* vector = history.index<float>({batch, beam, t, 0});
      attention.emplace_back(vector, vector + source_length);
    }
    return attention;
  }

  static std::vector<StorageView> build_logits(const StorageView& history,
                                                  const dim_t batch) {
    if (!history)
      return {};
    std::vector<StorageView> logits;
    logits.reserve(batch);
    for (dim_t t = 0; t < batch; ++t) {
      ops::Slide slide(0, t, 1);
      StorageView tmp(history.dtype(), history.device());
      slide(history, tmp);
      logits.emplace_back(std::move(tmp.squeeze(0)));
    }

    return logits;
  }

  static float compute_coverage_penalty(const std::vector<std::vector<float>>& attention,
                                        const float beta) {
    float penalty = 0;
    for (size_t column = 0; column < attention[0].size(); column++) {
      float coverage = 0;
      for (size_t row = 0; row < attention.size(); row++)
        coverage += attention[row][column];
      if (coverage > 0)
        penalty += std::log(std::min(coverage, 1.f));
    }
    return beta * penalty;
  }

  static float finalize_hypothesis_score(float score,
                                         const float length,
                                         const float length_penalty,
                                         const float coverage_penalty,
                                         const std::vector<std::vector<float>>* attention) {
    score /= std::pow(length, length_penalty);

    if (coverage_penalty != 0) {
      if (!attention)
        throw std::runtime_error("The attention weights are required to apply the coverage penalty");
      score += compute_coverage_penalty(*attention, coverage_penalty);
    }

    return score;
  }

  // Sort hypotheses from best to worst score, in the limit of max_hypotheses.
  static inline void sort_hypotheses(DecodingResult& result,
                                     size_t max_hypotheses,
                                     bool keep_scores,
                                     bool keep_attention,
                                     bool keep_logits_vocab) {
    std::vector<size_t> idx(result.hypotheses.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&result](size_t i1, size_t i2) { return result.scores[i1] > result.scores[i2]; });

    if (max_hypotheses < idx.size())
      idx.resize(max_hypotheses);

    result.hypotheses = index_vector(result.hypotheses, idx);

    if (keep_scores)
      result.scores = index_vector(result.scores, idx);
    else
      result.scores.clear();

    if (keep_attention)
      result.attention = index_vector(result.attention, idx);
    else
      result.attention.clear();

    if (keep_logits_vocab)
      result.logits_vocab = index_vector(result.logits_vocab, idx);
    else
      result.logits_vocab.clear();
  }

  static inline void finalize_result(DecodingResult& result,
                                     const size_t max_hypotheses,
                                     const float length_penalty,
                                     const float coverage_penalty,
                                     const bool keep_scores,
                                     const bool keep_attention,
                                     const bool keep_logits_vocab,
                                     const dim_t length_offset) {
    for (size_t i = 0; i < result.scores.size(); ++i) {
      const auto* attention = result.attention.empty() ? nullptr : &result.attention[i];
      result.scores[i] = finalize_hypothesis_score(result.scores[i],
                                                   result.hypotheses[i].size() + length_offset,
                                                   length_penalty,
                                                   coverage_penalty,
                                                   attention);
    }

    sort_hypotheses(result, max_hypotheses, keep_scores, keep_attention, keep_logits_vocab);
  }

  static inline size_t get_max_candidates(const dim_t beam_size, const float patience) {
    return std::round(float(beam_size) * patience);
  }

  static dim_t get_max_step(const dim_t max_length,
                            const bool return_prefix,
                            const std::vector<std::vector<size_t>>* prefix_ids) {
    dim_t max_step = 0;

    if (prefix_ids && !return_prefix) {
      for (const auto& ids : *prefix_ids) {
        const dim_t prefix_length = ids.size();
        max_step = std::max(max_step, prefix_length + max_length);
      }

    } else {
      max_step = max_length;
    }

    return max_step;
  }

  static inline bool is_last_step(const dim_t step,
                                  const dim_t max_length,
                                  const dim_t prefix_length,
                                  const bool return_prefix) {
    return step + 1 == max_length + (return_prefix ? 0 : prefix_length);
  }

  static void apply_min_length(const dim_t step,
                               const dim_t min_length,
                               const std::vector<size_t>& end_ids,
                               DisableTokens& disable_tokens,
                               const std::vector<dim_t>& batch_offset,
                               const bool return_prefix,
                               const std::vector<std::vector<size_t>>* prefix_ids) {
    if (prefix_ids && !return_prefix) {
      const size_t batch_size = batch_offset.size();

      for (size_t i = 0; i < batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        const dim_t prefix_length = prefix_ids->at(batch_id).size();

        if (step < prefix_length + min_length) {
          for (const size_t end_id : end_ids)
            disable_tokens.add(i, end_id);
        }
      }

    } else if (step < min_length) {
      for (const size_t end_id : end_ids)
        disable_tokens.add(end_id);
    }
  }


  BeamSearch::BeamSearch(const dim_t beam_size,
                         const float length_penalty,
                         const float coverage_penalty,
                         const float patience)
    : _beam_size(beam_size)
    , _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty)
    , _max_candidates(get_max_candidates(beam_size, patience))
  {
  }

  std::vector<DecodingResult>
  BeamSearch::search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     const Sampler& sampler,
                     const std::vector<size_t>& start_ids,
                     const std::vector<size_t>& end_ids,
                     const dim_t start_step,
                     const dim_t max_length,
                     const dim_t min_length,
                     const bool return_scores,
                     const bool return_attention,
                     const bool return_logits_vocab,
                     const bool return_prefix,
                     const size_t num_hypotheses,
                     const bool include_eos_in_hypotheses,
                     const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                     const std::vector<std::vector<size_t>>* prefix_ids) const {
    PROFILE("beam_search");
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const dim_t vocabulary_size = decoder.output_size();
    const dim_t batch_size = start_ids.size();

    // We get more candidates than the beam size so that if half the candidates are EOS,
    // we can replace finished hypotheses with active beams.
    const dim_t num_candidates = _beam_size * 2;

    // Only the first beam is considered in the first step. As an additional optimization
    // we try to run the first step without expanding the batch size.
    const bool expand_after_first_step = (device == Device::CPU
                                          && num_candidates <= vocabulary_size);

    // We can exit early when the first beam finishes and no penalties are used.
    const bool allow_early_exit = (_length_penalty == 0 && _coverage_penalty == 0);

    StorageView topk_ids({batch_size}, DataType::INT32);
    StorageView topk_scores(dtype);

    std::vector<bool> top_beam_finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    std::vector<DecodingResult> results(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      topk_ids.at<int32_t>(i) = start_ids[i];
    }

    if (!expand_after_first_step) {
      decoder.replicate_state(state, _beam_size);
      repeat_batch(topk_ids, _beam_size);
      TYPE_DISPATCH(dtype, initialize_beam_scores<T>(topk_scores, batch_size, _beam_size));
    }

    const bool use_hard_prefix = prefix_ids != nullptr;

    StorageView logits(dtype, device);
    StorageView alive_seq(topk_ids.dtype());
    StorageView alive_attention;

    const dim_t max_step = get_max_step(max_length,
                                        return_prefix,
                                        use_hard_prefix ? prefix_ids : nullptr);

    for (dim_t step = 0; step < max_step; ++step) {
      const bool is_expanded = (!expand_after_first_step || step > 0);

      // Compute log probs for the current step.
      StorageView attention_step(dtype, device);
      convert_to_original_word_ids(decoder, topk_ids);
      decoder(start_step + step,
              topk_ids.to(device),
              state,
              &logits,  // output shape: (cur_batch_size*beam_size x vocab_size), if not expanded beam_size is 1
              (return_attention || _coverage_penalty != 0) ? &attention_step : nullptr);

      const dim_t cur_batch_size = is_expanded ? logits.dim(0) / _beam_size : logits.dim(0);

      DisableTokens disable_tokens(logits);

      // Prevent the generation of end_ids until the minimum length is reached.
      apply_min_length(step,
                       min_length,
                       end_ids,
                       disable_tokens,
                       batch_offset,
                       return_prefix,
                       prefix_ids);

      if (!logits_processors.empty()) {
        if (alive_seq)
          merge_batch_beam(alive_seq);
        for (const auto& logits_processor : logits_processors)
          logits_processor->apply(step, logits, disable_tokens, alive_seq, batch_offset, prefix_ids);
        if (alive_seq)
          split_batch_beam(alive_seq, _beam_size);
      }

      disable_tokens.apply();
      std::vector<StorageView> logits_vec;
      if (return_logits_vocab) {
        if (is_expanded)
          logits_vec = build_logits(logits, cur_batch_size * _beam_size);
        else
          logits_vec = build_logits(logits, cur_batch_size);
      }

      StorageView log_probs(dtype, device);
      ops::LogSoftMax()(logits);
      log_probs.shallow_copy(logits);

      // Multiply by the current beam log probs.
      if (topk_scores) {
        DEVICE_AND_TYPE_DISPATCH(log_probs.device(), log_probs.dtype(),
                                 primitives<D>::add_depth_broadcast(topk_scores.to(device).data<T>(),
                                                                    log_probs.data<T>(),
                                                                    topk_scores.size(),
                                                                    log_probs.size()));
      }

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, -1});

      // TopK candidates.
      sampler(log_probs, topk_ids, topk_scores, num_candidates);

      // Unflatten the ids.
      StorageView gather_indices = unflatten_ids(topk_ids, _beam_size, vocabulary_size, is_expanded);

      if (use_hard_prefix) {
        update_sample_with_prefix(step,
                                  topk_ids,
                                  topk_scores,
                                  *prefix_ids,
                                  end_ids,
                                  batch_offset,
                                  _beam_size,
                                  &gather_indices,
                                  is_expanded);
      }

      // Append last prediction.
      append_step_output(alive_seq, topk_ids, &gather_indices);

      if (attention_step) {
        if (!is_expanded)
          repeat_batch(attention_step, _beam_size);
        split_batch_beam(attention_step, _beam_size);
        append_step_output(alive_attention, attention_step.to_float32().to(Device::CPU));
        gather_beam_flat(alive_attention, gather_indices, num_candidates);
      }

      // Check if some hypotheses are finished.
      std::vector<int32_t> non_finished_index;
      non_finished_index.reserve(cur_batch_size);

      // Only keep the first beam_size candidates.
      StorageView active_beams({cur_batch_size * _beam_size}, DataType::INT32);

      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        const dim_t prefix_length = use_hard_prefix ? prefix_ids->at(batch_id).size() : 0;
        const bool is_last_step_for_batch = is_last_step(step,
                                                         max_length,
                                                         prefix_length,
                                                         return_prefix);

        auto& result = results[batch_id];
        dim_t secondary_candidates_offset = _beam_size;

        for (dim_t k = 0; k < _beam_size; ++k) {
          const size_t last_id = topk_ids.at<int32_t>({i, k});
          dim_t next_beam_id = k;

          if ((is_eos(last_id, end_ids) && step >= prefix_length) || is_last_step_for_batch) {
            if (k == 0)
              top_beam_finished[i] = true;

            const bool ignore_last_token = is_eos(last_id, end_ids) && !include_eos_in_hypotheses;
            const dim_t start = return_prefix ? 0 : prefix_length;
            const dim_t end = ignore_last_token ? step : step + 1;

            // Register this hypothesis.
            result.scores.emplace_back(topk_scores.scalar_at<float>({i, k}));
            result.hypotheses.emplace_back(build_hypothesis(alive_seq, i, k, start, end));
            if (alive_attention)
              result.attention.emplace_back(build_attention(alive_attention, i, k, start, end));
            if (return_logits_vocab) {
              result.logits_vocab.emplace_back(std::move(logits_vec[i * k]));
            }

            // Move another active beam to this position.
            for (dim_t j = secondary_candidates_offset; j < num_candidates; ++j) {
              const auto candidate = topk_ids.at<int32_t>({i, j});
              if (!is_eos(candidate, end_ids)) {
                next_beam_id = j;
                secondary_candidates_offset = j + 1;
                break;
              }
            }
          }

          active_beams.at<int32_t>(i * _beam_size + k) = i * num_candidates + next_beam_id;
        }

        bool is_finished = false;
        if (is_last_step_for_batch)
          is_finished = true;
        else if (allow_early_exit)
          is_finished = top_beam_finished[i] && result.hypotheses.size() >= num_hypotheses;
        else
          is_finished = result.hypotheses.size() >= _max_candidates;

        if (is_finished) {
          finalize_result(result,
                          num_hypotheses,
                          _length_penalty,
                          _coverage_penalty,
                          return_scores,
                          return_attention,
                          return_logits_vocab,
                          _hypothesis_length_offset);
        } else {
          non_finished_index.emplace_back(i);
        }
      }

      const dim_t next_batch_size = non_finished_index.size();

      // If all remaining sentences are finished, no need to go further.
      if (next_batch_size == 0) {
        if (!is_expanded) {
          // We should ensure that states are replicated before exiting this function.
          decoder.replicate_state(state, _beam_size);
        }
        break;
      }

      gather(gather_indices, active_beams);
      gather_beam_flat(topk_ids, active_beams, _beam_size);
      gather_beam_flat(topk_scores, active_beams, _beam_size);
      gather_beam_flat(alive_seq, active_beams, _beam_size);
      if (alive_attention)
        gather_beam_flat(alive_attention, active_beams, _beam_size);

      // If some sentences finished on this step, ignore them for the next step.
      std::unique_ptr<StorageView> keep_batches;
      if (next_batch_size != cur_batch_size) {
        batch_offset = index_vector(batch_offset, non_finished_index);
        top_beam_finished = index_vector(top_beam_finished, non_finished_index);

        keep_batches = std::make_unique<StorageView>(Shape{next_batch_size}, non_finished_index);
        gather(topk_ids, *keep_batches);
        gather(topk_scores, *keep_batches);
        gather(alive_seq, *keep_batches);
        if (alive_attention)
          gather(alive_attention, *keep_batches);
        if (keep_batches->device() != device)
          *keep_batches = keep_batches->to(device);
      }

      if (gather_indices.device() != device)
        gather_indices = gather_indices.to(device);
      decoder.update_state(state, gather_indices, _beam_size, keep_batches.get());

      topk_ids.reshape({next_batch_size * _beam_size});
      topk_scores.reshape({next_batch_size * _beam_size});
    }

    return results;
  }


  GreedySearch::GreedySearch(const float length_penalty,
                             const float coverage_penalty)
    : _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty)
  {
  }

  std::vector<DecodingResult>
  GreedySearch::search(layers::Decoder& decoder,
                       layers::DecoderState& state,
                       const Sampler& sampler,
                       const std::vector<size_t>& start_ids,
                       const std::vector<size_t>& end_ids,
                       const dim_t start_step,
                       const dim_t max_length,
                       const dim_t min_length,
                       const bool return_scores,
                       const bool return_attention,
                       const bool return_logits_vocab,
                       const bool return_prefix,
                       const size_t num_hypotheses,
                       const bool include_eos_in_hypotheses,
                       const std::vector<std::shared_ptr<LogitsProcessor>>& logits_processors,
                       const std::vector<std::vector<size_t>>* prefix_ids) const {
    const dim_t batch_size = start_ids.size();

    // We can return multiple hypotheses from greedy search when random sampling is enabled.
    // In that case we replicate the batches and then merge the hypotheses in a single result.
    if (num_hypotheses > 1) {
      for (auto& [name, value] : state) {
        if (value)
          repeat_batch(value, num_hypotheses);
      }

      std::vector<size_t> repeat_start_ids = repeat_vector(start_ids, num_hypotheses);
      std::vector<std::vector<size_t>> repeat_prefix_ids;
      if (prefix_ids)
        repeat_prefix_ids = repeat_vector(*prefix_ids, num_hypotheses);

      std::vector<DecodingResult> results = search(
        decoder,
        state,
        sampler,
        repeat_start_ids,
        end_ids,
        start_step,
        max_length,
        min_length,
        /*return_scores=*/true,
        return_attention,
        return_logits_vocab,
        return_prefix,
        /*num_hypotheses=*/1,
        include_eos_in_hypotheses,
        logits_processors,
        prefix_ids ? &repeat_prefix_ids : nullptr);

      std::vector<DecodingResult> final_results(batch_size);

      for (size_t i = 0; i < results.size(); ++i) {
        auto& result = results[i];
        auto& final_result = final_results[i / num_hypotheses];

        final_result.hypotheses.emplace_back(std::move(result.hypotheses[0]));
        final_result.scores.emplace_back(result.scores[0]);
        if (return_attention)
          final_result.attention.emplace_back(std::move(result.attention[0]));
        if (return_logits_vocab)
          final_result.logits_vocab.emplace_back(std::move(result.logits_vocab[0]));
      }

      for (auto& result : final_results)
        sort_hypotheses(result, num_hypotheses, return_scores, return_attention, return_logits_vocab);

      return final_results;
    }

    PROFILE("greedy_search");
    const Device device = decoder.device();
    const DataType dtype = decoder.output_type();
    const bool gather_attention = (return_attention || (return_scores && _coverage_penalty != 0));

    StorageView sample_from({batch_size}, DataType::INT32);

    StorageView logits(dtype, device);
    std::vector<dim_t> batch_offset(batch_size);
    std::vector<DecodingResult> results(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sample_from.at<int32_t>(i) = start_ids[i];
      results[i].hypotheses.resize(1);
      if (return_scores)
        results[i].scores.resize(1, 0.f);
      if (return_attention)
        results[i].attention.resize(1);
    }

    StorageView best_ids(DataType::INT32);
    StorageView best_probs(dtype);
    StorageView alive_seq(DataType::INT32);
    StorageView attention_step;
    StorageView attention_step_device(dtype, device);

    const dim_t max_step = get_max_step(max_length, return_prefix, prefix_ids);
    dim_t first_step = 0;

    // Device fast path (CUDA graphs): the token selection runs on the device, inside the
    // graph of the decoder step, and for a single sequence several steps are queued
    // before their results are read. Only for configurations where the selection is the
    // exact device equivalent of the loop below: argmax sampling, token suppression as
    // the only logits processing, no prefix to force, no per-step outputs. The results
    // are processed by the same code as below, step by step; steps queued past the end
    // of the sequence are discarded.
    layers::Decoder::GreedyStepOptions device_options;
    const dim_t max_queued = decoder.max_queued_greedy_steps();
    bool use_device_steps = (max_queued > 0
                             && dynamic_cast<const BestSampler*>(&sampler) != nullptr
                             && !gather_attention
                             && !return_logits_vocab
                             && min_length == 0);
    if (use_device_steps && prefix_ids) {
      for (const auto& prefix : *prefix_ids)
        use_device_steps = use_device_steps && prefix.empty();
    }
    for (const auto& processor : logits_processors) {
      if (!use_device_steps)
        break;
      if (const auto* suppress = dynamic_cast<const SuppressTokens*>(processor.get())) {
        device_options.disable_ids.insert(device_options.disable_ids.end(),
                                          suppress->ids().begin(),
                                          suppress->ids().end());
      } else if (const auto* suppress_begin =
                 dynamic_cast<const SuppressTokensBegin*>(processor.get())) {
        device_options.disable_ids_begin.insert(device_options.disable_ids_begin.end(),
                                                suppress_begin->ids().begin(),
                                                suppress_begin->ids().end());
      } else {
        use_device_steps = false;
      }
    }

    if (use_device_steps) {
      device_options.disable_value = std::numeric_limits<float>::lowest();  // DisableTokens.
      device_options.log_softmax = return_scores;

      std::vector<int32_t> input_ids;
      std::vector<int32_t> step_ids;
      std::vector<float> step_scores;
      bool feed_ids = true;

      while (first_step < max_step) {
        const dim_t cur_batch_size = batch_offset.size();
        const dim_t num_steps = (cur_batch_size == 1
                                 ? std::min(max_queued, max_step - first_step)
                                 : 1);
        if (feed_ids) {
          input_ids.resize(cur_batch_size);
          for (dim_t i = 0; i < cur_batch_size; ++i)
            input_ids[i] = decoder.to_original_word_id(sample_from.at<int32_t>(i));
        }

        dim_t queued = 0;
        for (; queued < num_steps; ++queued) {
          // SuppressTokensBegin applies at the first step (no prefix).
          const dim_t step = first_step + queued;
          if (!decoder.queue_greedy_step(start_step + step,
                                         device_options,
                                         queued == 0 && feed_ids ? &input_ids : nullptr,
                                         /*begin_step=*/step == 0,
                                         state))
            break;
        }
        if (queued == 0)
          break;  // Continue with the regular loop.
        feed_ids = false;
        decoder.read_greedy_steps(start_step + first_step, queued, step_ids, step_scores);

        bool batch_changed = false;
        for (dim_t k = 0; k < queued && !batch_changed; ++k) {
          const dim_t step = first_step;
          StorageView best_ids({cur_batch_size, 1}, DataType::INT32);
          for (dim_t i = 0; i < cur_batch_size; ++i)
            best_ids.at<int32_t>(i) = step_ids[k * cur_batch_size + i];

          if (!logits_processors.empty()) {
            if (alive_seq) {
              const StorageView cur_alive_seq = std::move(alive_seq);
              ops::Concat(-1)({&cur_alive_seq, &best_ids}, alive_seq);
            } else {
              alive_seq = best_ids;
            }
          }

          std::vector<int32_t> non_finished_index;
          non_finished_index.reserve(cur_batch_size);

          for (dim_t i = 0; i < cur_batch_size; ++i) {
            const size_t word_id = best_ids.at<int32_t>(i);
            const size_t batch_id = batch_offset[i];
            const dim_t prefix_length = prefix_ids ? prefix_ids->at(batch_id).size() : 0;
            const float score = step_scores[k * cur_batch_size + i];

            if ((!is_eos(word_id, end_ids) || include_eos_in_hypotheses)
                && (return_prefix || step >= prefix_length)) {
              results[batch_id].hypotheses[0].push_back(word_id);
            }

            if (return_scores)
              results[batch_id].scores[0] += score;

            const bool is_finished = ((is_eos(word_id, end_ids) && step >= prefix_length)
                                      || (is_last_step(step, max_length, prefix_length, return_prefix)));

            if (is_finished) {
              finalize_result(results[batch_id],
                              1,
                              _length_penalty,
                              _coverage_penalty,
                              return_scores,
                              return_attention,
                              return_logits_vocab,
                              _hypothesis_length_offset);
            } else {
              non_finished_index.emplace_back(i);
              sample_from.at<int32_t>(i) = word_id;
            }
          }

          ++first_step;
          const dim_t count_alive = non_finished_index.size();

          // No more sentences are alive, stop here.
          if (count_alive == 0)
            return results;

          // Remove finished sentences from the execution.
          if (count_alive != cur_batch_size) {
            batch_offset = index_vector(batch_offset, non_finished_index);

            StorageView alive({count_alive}, non_finished_index);
            if (alive_seq)
              gather(alive_seq, alive);
            gather(sample_from, alive);
            decoder.update_state(state, alive.to(device));
            batch_changed = true;
            feed_ids = true;
          }
        }
      }
    }

    for (dim_t step = first_step; step < max_step; ++step) {
      convert_to_original_word_ids(decoder, sample_from);
      decoder(start_step + step,
              sample_from.to(device),
              state,
              &logits,
              gather_attention ? &attention_step_device : nullptr);

      DisableTokens disable_tokens(logits);

      // Prevent the generation of end_id until the minimum length is reached.
      apply_min_length(step,
                       min_length,
                       end_ids,
                       disable_tokens,
                       batch_offset,
                       return_prefix,
                       prefix_ids);

      for (const auto& logits_processor : logits_processors)
        logits_processor->apply(step, logits, disable_tokens, alive_seq, batch_offset, prefix_ids);

      disable_tokens.apply();

      std::vector<StorageView> logits_vec;
      StorageView logits_orig(dtype, device);
      if (return_logits_vocab) {
        logits_vec = build_logits(logits, logits.dim(0));
        logits_orig.copy_from(logits);
      }
      // Compute log probs only if required.
      StorageView log_probs(dtype, device);
      if (return_scores)
        ops::LogSoftMax()(logits);
      log_probs.shallow_copy(logits);

      sampler(log_probs, best_ids, best_probs);
      if (prefix_ids)
        update_sample_with_prefix(step, best_ids, best_probs, *prefix_ids, end_ids, batch_offset);
      if (attention_step_device)
        attention_step.copy_from(attention_step_device.to_float32());

      if (!logits_processors.empty()) {
        if (alive_seq) {
          const StorageView cur_alive_seq = std::move(alive_seq);
          ops::Concat(-1)({&cur_alive_seq, &best_ids}, alive_seq);
        } else {
          alive_seq = best_ids;
        }
      }

      const dim_t cur_batch_size = log_probs.dim(0);
      std::vector<int32_t> non_finished_index;
      non_finished_index.reserve(cur_batch_size);

      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const size_t word_id = best_ids.at<int32_t>(i);
        const size_t batch_id = batch_offset[i];
        const dim_t prefix_length = prefix_ids ? prefix_ids->at(batch_id).size() : 0;
        const float score = best_probs.scalar_at<float>({i, 0});

        if (return_logits_vocab) {
          results[batch_id].logits_vocab.resize(1);
          results[batch_id].logits_vocab[0].emplace_back(std::move(logits_vec[i]));
        }

        if ((!is_eos(word_id, end_ids) || include_eos_in_hypotheses)
            && (return_prefix || step >= prefix_length)) {
          results[batch_id].hypotheses[0].push_back(word_id);
          if (attention_step) {
            const auto* attn = attention_step.index<float>({i, 0});
            results[batch_id].attention[0].emplace_back(attn, attn + attention_step.dim(-1));
          }
        }

        if (return_scores)
          results[batch_id].scores[0] += score;

        bool is_finished = ((is_eos(word_id, end_ids) && step >= prefix_length)
                            || (is_last_step(step, max_length, prefix_length, return_prefix)));

        if (is_finished) {
          finalize_result(results[batch_id],
                          1,
                          _length_penalty,
                          _coverage_penalty,
                          return_scores,
                          return_attention,
                          return_logits_vocab,
                          _hypothesis_length_offset);
        } else {
          non_finished_index.emplace_back(i);
          sample_from.at<int32_t>(i) = word_id;
        }
      }

      const dim_t count_alive = non_finished_index.size();

      // No more sentences are alive, stop here.
      if (count_alive == 0)
        break;

      // Remove finished sentences from the execution.
      if (count_alive != cur_batch_size) {
        batch_offset = index_vector(batch_offset, non_finished_index);

        StorageView alive({count_alive}, non_finished_index);
        if (alive_seq)
          gather(alive_seq, alive);
        gather(sample_from, alive);
        decoder.update_state(state, alive.to(device));
      }
    }

    return results;
  }

  static std::pair<std::vector<size_t>, std::vector<std::vector<size_t>>>
  split_start_tokens(const std::vector<std::vector<size_t>>& start_tokens) {
    std::vector<size_t> start_ids;
    std::vector<std::vector<size_t>> prefix_ids;
    start_ids.reserve(start_tokens.size());
    prefix_ids.reserve(start_tokens.size());
    bool only_start_token = true;

    for (const auto& tokens : start_tokens) {
      if (tokens.empty())
        throw std::invalid_argument("One input has no decoder start token");
      if (tokens.size() > 1)
        only_start_token = false;

      start_ids.emplace_back(tokens.front());
      prefix_ids.emplace_back(tokens.begin() + 1, tokens.end());
    }

    if (only_start_token)
      prefix_ids.clear();

    return std::make_pair(std::move(start_ids), std::move(prefix_ids));
  }

  static void validate_decoding_options(const DecodingOptions& options, const Device device) {
    if (options.beam_size == 0)
      throw std::invalid_argument("The beam size must be > 0");
    if (options.patience <= 0)
      throw std::invalid_argument("The patience factor must be > 0");
    if (options.num_hypotheses == 0)
      throw std::invalid_argument("The number of hypotheses must be > 0");
    if (options.num_hypotheses > get_max_candidates(options.beam_size, options.patience)
        && !(options.beam_size == 1 && options.sampling_topk != 1))
      throw std::invalid_argument("The number of hypotheses cannot be greater than "
                                  "beam_size * patience");
    if (options.min_length > options.max_length)
      throw std::invalid_argument("The minimum decoding length is greater than "
                                  "the maximum decoding length");
    if (options.max_length == 0)
      throw std::invalid_argument("The maximum decoding length must be > 0");
    if (options.repetition_penalty <= 0)
      throw std::invalid_argument("The repetition penalty must be > 0");

    if (options.sampling_topp <= 0 || options.sampling_topp > 1)
      throw std::invalid_argument("The sampling_topp parameter must be between 0 and 1");
    if (options.sampling_topp < 1
        && options.sampling_topk > static_cast<size_t>(ops::TopPMask::max_num_classes(device)))
      throw std::invalid_argument(
        "The sampling_topp parameter currently requires sampling_topk <= "
        + std::to_string(ops::TopPMask::max_num_classes(device))
        + " when running on a " + device_to_str(device) + " device");
  }

  static std::unique_ptr<const Sampler>
  make_sampler(const DecodingOptions& options) {
    if (options.sampling_topk == 1 || options.sampling_temperature == 0.0)
      return std::make_unique<BestSampler>();
    else
      return std::make_unique<RandomSampler>(options.sampling_topk,
                                             options.sampling_topp,
                                             options.sampling_temperature);
  }

  static std::unique_ptr<const SearchStrategy>
  make_search_strategy(const DecodingOptions& options) {
    std::unique_ptr<SearchStrategy> strategy;
    if (options.beam_size == 1)
      strategy = std::make_unique<GreedySearch>(options.length_penalty,
                                                options.coverage_penalty);
    else
      strategy = std::make_unique<BeamSearch>(options.beam_size,
                                              options.length_penalty,
                                              options.coverage_penalty,
                                              options.patience);
    strategy->set_hypothesis_length_offset(options.folded_prefix_length);
    return strategy;
  }

  static std::vector<std::shared_ptr<LogitsProcessor>>
  make_logits_processors(const DecodingOptions& options) {
    std::vector<std::shared_ptr<LogitsProcessor>> processors;

    for (const auto& processor : options.logits_processors) {
      if (processor->apply_first())
        processors.emplace_back(processor);
    }

    if (options.repetition_penalty != 1)
      processors.emplace_back(std::make_shared<RepetitionPenalty>(options.repetition_penalty));

    if (options.no_repeat_ngram_size > 0)
      processors.emplace_back(std::make_shared<NoRepeatNgram>(options.no_repeat_ngram_size));

    if (!options.disable_ids.empty())
      processors.emplace_back(std::make_shared<SuppressTokens>(options.disable_ids));

    if (!options.disable_ids_begin.empty())
      processors.emplace_back(std::make_shared<SuppressTokensBegin>(options.disable_ids_begin));


    for (const auto& processor : options.logits_processors) {
      if (!processor->apply_first())
        processors.emplace_back(processor);
    }

    return processors;
  }

  static std::vector<size_t> map_to_output_word_ids(const layers::Decoder& decoder,
                                                    const std::vector<size_t>& ids) {
    std::vector<size_t> new_ids;
    new_ids.reserve(ids.size());
    for (const size_t id : ids) {
      if (decoder.is_in_output(id))
        new_ids.push_back(decoder.to_output_word_id(id));
    }
    return new_ids;
  }

  std::vector<DecodingResult>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         std::vector<std::vector<size_t>> start_tokens,
         std::vector<size_t> end_ids,
         DecodingOptions options) {
    validate_decoding_options(options, decoder.device());
    const size_t batch_size = start_tokens.size();

    if (batch_size == 0)
      throw std::invalid_argument("No decoder start tokens are set");

    std::vector<DecodingResult> results;

    if (decoder.output_layer_is_updated()) {
      end_ids = map_to_output_word_ids(decoder, end_ids);

      for (auto& ids : start_tokens)
        ids = map_to_output_word_ids(decoder, ids);

      options.disable_ids = map_to_output_word_ids(decoder, options.disable_ids);
      options.disable_ids_begin = map_to_output_word_ids(decoder, options.disable_ids_begin);
    }

    std::vector<size_t> start_ids;
    std::vector<std::vector<size_t>> prefix_ids;
    std::tie(start_ids, prefix_ids) = split_start_tokens(start_tokens);
    // A folded prefix ended right before the start token: keep the search in prefix
    // mode (empty prefixes) so the first free step is treated as after a forced prefix.
    if (options.folded_prefix_length > 0 && prefix_ids.empty())
      prefix_ids.assign(batch_size, {});

    const auto search_strategy = make_search_strategy(options);
    const auto sampler = make_sampler(options);
    const auto logits_processors = make_logits_processors(options);
    results = search_strategy->search(decoder,
                                      state,
                                      *sampler,
                                      start_ids,
                                      end_ids,
                                      options.start_step,
                                      options.max_length,
                                      options.min_length,
                                      options.return_scores,
                                      options.return_attention,
                                      options.return_logits_vocab,
                                      options.return_prefix,
                                      options.num_hypotheses,
                                      options.include_eos_in_hypotheses,
                                      logits_processors,
                                      prefix_ids.empty() ? nullptr : &prefix_ids);

    if (decoder.output_layer_is_updated()) {
      // Restore original word ids.
      for (auto& result : results) {
        for (auto& hypothesis : result.hypotheses) {
          for (auto& id : hypothesis)
            id = decoder.to_original_word_id(id);
        }
      }
    }

    return results;
  }

}
