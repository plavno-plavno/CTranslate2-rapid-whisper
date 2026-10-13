#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/padder.h"

namespace ctranslate2 {
  namespace layers {

    // While a decoder step is captured into a CUDA graph, the self-attention of the
    // current layer concatenates its cache into these persistent buffers instead of a
    // fresh allocation, so that the graph replays against stable addresses. The layout
    // and the copy are the same as the regular concatenation. Thread-local, null when
    // no capture is in progress.
    struct SelfAttentionCacheTarget {
      StorageView* keys;
      StorageView* values;
    };
    SelfAttentionCacheTarget*& self_attention_cache_target();
    StorageView make_relative_positions(dim_t queries_length,
                                        dim_t keys_length,
                                        dim_t max_position);


    class AttentionLayer : public Layer
    {
    public:
      AttentionLayer(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false);
      virtual ~AttentionLayer() {};
      DataType output_type() const override;
      dim_t output_size() const override;
      virtual void operator()(const StorageView& queries,
                      const StorageView& values,
                      const StorageView* values_lengths,
                      StorageView& output,
                      StorageView* cached_keys = nullptr,
                      StorageView* cached_values = nullptr,
                      StorageView* attention = nullptr,
                      const Padder* queries_padder = nullptr,
                      const Padder* values_padder = nullptr,
                      bool return_normalized_attention = true,
                      StorageView* position_bias = nullptr,
                      dim_t offset = 0) const = 0;

      virtual bool has_positional_embeddings() const = 0;

      bool multi_query() const {
        return _multi_query;
      }

      static StorageView prepare_length_mask(const StorageView& lengths,
                                             const dim_t num_heads,
                                             const dim_t num_queries,
                                             const bool mask_future = false,
                                             const bool multi_query = false);

    protected:
      const dim_t _num_heads;
      const bool _self_attention;
      const bool _is_decoder;
      const std::vector<Dense> _linear;
      const dim_t _d_model;
      const dim_t _d_head;
      const bool _pre_norm;
      const std::unique_ptr<const LayerNorm> _layer_norm;
      const float _queries_scale;
      const bool _multi_query;
      const dim_t _num_heads_kv;
      const dim_t _sliding_window;
    };
  }
}
