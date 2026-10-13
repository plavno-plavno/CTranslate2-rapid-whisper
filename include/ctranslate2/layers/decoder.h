#pragma once

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ctranslate2/layers/common.h"
#include "ctranslate2/storage_view.h"

namespace ctranslate2 {
  namespace layers {

    using DecoderState = std::unordered_map<std::string, StorageView>;

    void zero_first_timestep(StorageView& x, dim_t step);

    // Base class for decoders.
    class Decoder : public Layer {
    public:
      Decoder(Device device);

      virtual DecoderState initial_state(bool iterative_decoding = true) const = 0;

      // Forwards one step.
      virtual void operator()(dim_t step,
                              const StorageView& ids,
                              DecoderState& state,
                              StorageView* logits = nullptr,
                              StorageView* attention = nullptr) = 0;

      // Forwards a full sequence.
      virtual void operator()(const StorageView& ids,
                              const StorageView& lengths,
                              DecoderState& state,
                              StorageView& logits,
                              StorageView* attention = nullptr) = 0;

      // Update the decoder state in greedy search.
      void update_state(DecoderState& state, const StorageView& alive_batches) const;

      // Update the decoder state in beam search.
      void update_state(DecoderState& state,
                        StorageView beam_indices,
                        const dim_t beam_size,
                        const StorageView* alive_batches = nullptr) const;

      // Replicate the decoder state beam_size times.
      void replicate_state(DecoderState& state, const dim_t beam_size) const;

      // Returns true if the state must be replicated beam_size times.
      virtual bool replicate_state(const std::string& name) const;

      // Incremented each time update_output_layer replaces the output weights.
      dim_t output_layer_version() const {
        return _output_layer_version;
      }

      // Restrict the output layer to a set of ids and/or resize it to a preferred size multiple.
      // Elements in restrict_ids must be unique and sorted.
      void update_output_layer(const dim_t size_multiple = 1,
                               const std::vector<size_t>& restrict_ids = {});

      bool output_layer_is_updated() const {
        return !_to_original_word_id.empty();
      }

      bool is_in_output(size_t word_id) const {
        return _to_output_word_id.find(word_id) != _to_output_word_id.end();
      }

      size_t to_output_word_id(size_t original_id) const {
        return _to_output_word_id.empty() ? original_id : _to_output_word_id.at(original_id);
      }

      size_t to_original_word_id(size_t output_id) const {
        return _to_original_word_id.empty() ? output_id : _to_original_word_id.at(output_id);
      }

      Device device() const {
        return _device;
      }

      DataType output_type() const override {
        return const_cast<Decoder&>(*this).output_layer().output_type();
      }

      dim_t output_size() const override {
        return const_cast<Decoder&>(*this).output_layer().output_size();
      }

      // Greedy decoding step that stays on the device: forwards `step`, applies the
      // token suppression, the optional log softmax and the argmax exactly as the greedy
      // search does, and feeds the selected ids to the next step without a host round
      // trip. Steps can be queued back to back and their results read in one go.
      struct GreedyStepOptions {
        std::vector<size_t> disable_ids;        // Output vocabulary ids, every step.
        std::vector<size_t> disable_ids_begin;  // Output vocabulary ids, begin step only.
        float disable_value = 0;
        bool log_softmax = false;

        bool operator==(const GreedyStepOptions& other) const {
          return disable_ids == other.disable_ids
            && disable_ids_begin == other.disable_ids_begin
            && disable_value == other.disable_value
            && log_softmax == other.log_softmax;
        }
      };

      // Number of greedy steps that may be queued before reading their results
      // (0 when device greedy steps are not supported).
      virtual dim_t max_queued_greedy_steps() const {
        return 0;
      }

      // Queues one step. `input_ids` (original vocabulary ids, one per batch) must be set
      // for the first step and after the batch changed; otherwise the step consumes the
      // ids selected by the previous queued step. Returns false when the step cannot run
      // on the device: nothing was queued and the caller must decode it regularly.
      virtual bool queue_greedy_step(dim_t,
                                     const GreedyStepOptions&,
                                     const std::vector<int32_t>*,
                                     bool,
                                     DecoderState&) {
        return false;
      }

      // Waits for the queued steps and returns their selected output ids and scores,
      // step-major ([num_steps][batch_size]).
      virtual void read_greedy_steps(dim_t,
                                     dim_t,
                                     std::vector<int32_t>&,
                                     std::vector<float>&) {
      }

    protected:
      // Returns the current batch size from the decoder state.
      virtual dim_t batch_size(const DecoderState& state) const;
      // Returns the output linear layer.
      virtual Dense& output_layer() = 0;

      const Device _device;

    private:
      std::vector<size_t> _to_original_word_id;
      std::unordered_map<size_t, size_t> _to_output_word_id;
      dim_t _vocabulary_size = 0;
      dim_t _output_layer_version = 0;
    };


    class DecoderStateCache {
    public:
      void save(std::vector<size_t> prompt, DecoderState state);
      const DecoderState* get(const std::vector<size_t>& prompt) const;

    private:
      std::map<std::vector<size_t>, DecoderState> _cache;
      mutable std::mutex _mutex;
    };

  }
}
