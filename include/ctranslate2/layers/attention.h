#pragma once

#include "ctranslate2/layers/attention_layer.h"
#include "ctranslate2/padder.h"
#include "ctranslate2/layers/transformer.h"

namespace ctranslate2 {
  namespace layers {

    // Standard multi-head attention, as used by Whisper: fused QKV (self-attention) or
    // Q + fused KV (cross-attention) projections, scaled dot product, output projection.
    class MultiHeadAttention : public AttentionLayer
    {
    public:
      MultiHeadAttention(const models::Model& model,
                         const std::string& scope,
                         dim_t num_heads,
                         bool self_attention,
                         bool pre_norm = true,
                         bool is_decoder = false);
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
                      bool return_normalized_attention = true) const override;

    protected:
      void process_cross_attention(const StorageView& values,
                                   StorageView& fused_proj,
                                   StorageView& queries_proj,
                                   StorageView& keys_proj,
                                   StorageView& values_proj,
                                   StorageView* cached_keys,
                                   StorageView* cached_values,
                                   const Padder* queries_padder,
                                   const Padder* values_padder,
                                   dim_t& beam_size) const;

    private:
      static void split_heads(StorageView& x,
                               dim_t num_heads,
                               const Padder* padder = nullptr,
                               dim_t beam_size = 1);

      static void combine_heads(StorageView& x,
                                 dim_t num_heads,
                                 const Padder* padder = nullptr,
                                 dim_t beam_size = 1);

      // The self-attention cache is [batch, heads, time, depth].
      static constexpr dim_t _cache_time_dim = 2;
    };
  }
}
