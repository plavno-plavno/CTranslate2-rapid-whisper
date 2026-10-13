#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/split.h"
#include "ctranslate2/utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "dispatch.h"

namespace ctranslate2 {
  namespace layers {

    SelfAttentionCacheTarget*& self_attention_cache_target() {
      static thread_local SelfAttentionCacheTarget* target = nullptr;
      return target;
    }

    static const ops::Transpose transpose_op({0, 2, 1, 3});

    static inline void save_attention(StorageView& attention, StorageView weights, dim_t beam_size) {
      if (beam_size == 1)
        attention = std::move(weights);
      else {
        transpose_op(weights, attention);
        attention.reshape({-1, weights.dim(1), 1, weights.dim(-1)});
      }
    }

    static void dot_product_attention(const StorageView& queries,
                                      const StorageView& keys,
                                      const StorageView& values,
                                      const StorageView* values_lengths,
                                      StorageView& output,
                                      StorageView* attention,
                                      bool return_normalized_attention,
                                      float queries_scale,
                                      dim_t beam_size) {
      PROFILE("dot_product_attention");

      const ops::MatMul keys_matmul(/*trans_a=*/false, /*trans_b=*/true, queries_scale);
      keys_matmul(queries, keys, output);

      StorageView attn(values.dtype(), values.device());
      if (attention && !return_normalized_attention) {
        ops::SoftMax()(output, values_lengths, attn);
        save_attention(*attention, std::move(output), beam_size);
      } else {
        attn = std::move(output);
        ops::SoftMax()(attn, values_lengths, attn);
      }

      const ops::MatMul values_matmul;
      values_matmul(attn, values, output);

      if (attention && return_normalized_attention)
        save_attention(*attention, std::move(attn), beam_size);
    }

    MultiHeadAttention::MultiHeadAttention(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder)
      : AttentionLayer(model, scope, num_heads, self_attention, pre_norm, is_decoder)
    {
    }

    DataType MultiHeadAttention::output_type() const {
      return _linear.back().output_type();
    }

    dim_t MultiHeadAttention::output_size() const {
      return _d_model;
    }

    void MultiHeadAttention::process_cross_attention(
        const StorageView& values,
        StorageView& fused_proj,
        StorageView& queries_proj,
        StorageView& keys_proj,
        StorageView& values_proj,
        StorageView* cached_keys,
        StorageView* cached_values,
        const Padder* queries_padder,
        const Padder* values_padder,
        dim_t& beam_size) const {

      queries_proj = std::move(fused_proj);

      if (cached_keys == nullptr || cached_keys->empty()) {
        _linear[1](values, fused_proj);

        split_heads(fused_proj, 2 * _num_heads, values_padder);
        ops::Split(1)(fused_proj, keys_proj, values_proj);

        if (cached_keys != nullptr) {
          *cached_keys = std::move(keys_proj);
          *cached_values = std::move(values_proj);
        }
      }

      if (queries_proj.dim(1) == 1 && cached_keys)
        beam_size = queries_proj.dim(0) / cached_keys->dim(0);

      split_heads(queries_proj, _num_heads, queries_padder, beam_size);
    }

    void MultiHeadAttention::operator()(const StorageView& queries,
                                        const StorageView& values,
                                        const StorageView* values_lengths,
                                        StorageView& output,
                                        StorageView* cached_keys,
                                        StorageView* cached_values,
                                        StorageView* attention,
                                        const Padder* queries_padder,
                                        const Padder* values_padder,
                                        bool return_normalized_attention) const {
      PROFILE("MultiHeadAttention");
      const Device device = queries.device();
      const DataType dtype = queries.dtype();
      StorageView fused_proj(dtype, device);
      StorageView queries_proj(dtype, device);
      StorageView keys_proj(dtype, device);
      StorageView values_proj(dtype, device);

      const StorageView* q = &queries;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(queries, queries_proj);
        q = &queries_proj;
      }

      _linear[0](*q, fused_proj);

      dim_t beam_size = 1;

      if (!_self_attention) {
        process_cross_attention(values, fused_proj, queries_proj, keys_proj,
                                values_proj, cached_keys, cached_values,
                                queries_padder, values_padder, beam_size);
      } else {
        split_heads(fused_proj, 3 * _num_heads, queries_padder);
        ops::Split(1)(fused_proj, queries_proj, keys_proj, values_proj);

        if (cached_keys != nullptr) {
          SelfAttentionCacheTarget* target = self_attention_cache_target();
          if (cached_keys->empty()) {
            *cached_keys = std::move(keys_proj);
            *cached_values = std::move(values_proj);
          } else if (target) {
            const ops::Concat concat_op(_cache_time_dim);
            concat_op({cached_keys, &keys_proj}, *target->keys);
            concat_op({cached_values, &values_proj}, *target->values);
            cached_keys->shallow_copy(*target->keys);
            cached_values->shallow_copy(*target->values);
          } else {
            const ops::Concat concat_op(_cache_time_dim);
            StorageView& tmp = fused_proj;  // Reuse storage.
            tmp = std::move(*cached_keys);
            concat_op({&tmp, &keys_proj}, *cached_keys);
            tmp = std::move(*cached_values);
            concat_op({&tmp, &values_proj}, *cached_values);
          }
        }
      }

      if (cached_keys) {
        keys_proj.shallow_copy(*cached_keys);
        values_proj.shallow_copy(*cached_values);
      }

      StorageView& context = fused_proj;  // Reuse storage.
      dot_product_attention(queries_proj,
                            keys_proj,
                            values_proj,
                            values_lengths,
                            context,
                            attention,
                            return_normalized_attention,
                            _queries_scale,
                            beam_size);

      combine_heads(context, _num_heads, queries_padder, beam_size);
      _linear.back()(context, output, _layer_norm ? &queries : nullptr);

      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }

    void MultiHeadAttention::split_heads(StorageView& x,
                     dim_t num_heads,
                     const Padder* padder,
                     dim_t beam_size) {
      if (padder)
        padder->add_padding(x);

      if (beam_size > 1)
        x.reshape({x.dim(0) / beam_size, beam_size, x.dim(2)});

      // x has shape [batch_size, time, depth]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(1);
      const dim_t head_dim = x.dim(2) / num_heads;

      if (time == 1) {
        x.reshape({batch_size, num_heads, 1, head_dim});
      } else {
        x.reshape({batch_size, time, num_heads, head_dim});
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }
    }

    void MultiHeadAttention::combine_heads(StorageView& x,
                                         dim_t num_heads,
                                         const Padder* padder,
                                         dim_t beam_size) {
      // x has shape [batch_size, num_heads, time, head_dim]
      const dim_t batch_size = x.dim(0);
      const dim_t time = x.dim(2);
      const dim_t depth = x.dim(3) * num_heads;

      if (time > 1) {
        StorageView y(x.device(), x.dtype());
        transpose_op(x, y);
        x = std::move(y);
      }

      x.reshape({batch_size, time, depth});

      if (beam_size > 1)
        x.reshape({batch_size * beam_size, 1, depth});

      if (padder)
        padder->remove_padding(x);
    }
  }
}
