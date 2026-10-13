#include "ctranslate2/layers/attention.h"
#include "ctranslate2/ops/split.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "dispatch.h"

namespace ctranslate2 {
  namespace layers {
    static std::vector<Dense> make_linear_layers(const models::Model& model,
                                                 const std::string& scope,
                                                 bool self_attention) {
      const dim_t num_linear_layers = self_attention ? 2 : 3;
      std::vector<Dense> layers;
      layers.reserve(num_linear_layers);
      for (dim_t i = 0; i < num_linear_layers; ++i)
        layers.emplace_back(model, scope + "/linear_" + std::to_string(i));
      return layers;
    }

    AttentionLayer::AttentionLayer(const models::Model& model,
                                           const std::string& scope,
                                           dim_t num_heads,
                                           bool self_attention,
                                           bool pre_norm,
                                           bool is_decoder)
      : _num_heads(num_heads)
      , _self_attention(self_attention)
      , _is_decoder(is_decoder)
      , _linear(make_linear_layers(model, scope, self_attention))
      , _d_model(_linear.back().output_size())
      , _d_head(model.get_attribute_with_default<int32_t >(scope + "/head_dim", _d_model / _num_heads))
      , _pre_norm(pre_norm)
      , _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _queries_scale(model.get_attribute_with_default<float>(
                         scope + "/queries_scale",
                         1.f / std::sqrt(static_cast<float>(_d_head))))
    {
      for (const char* option : {"/multi_query", "/num_heads_kv", "/sliding_window",
                                 "/rotary_dim", "/relative_attention_bias",
                                 "/relative_position_keys", "/q_norm/gamma", "/memory_kv/weight"}) {
        if (model.get_variable_if_exists(scope + option))
          throw std::invalid_argument("Unsupported attention option " + scope + option
                                      + " in this Whisper-only build");
      }
    }

    DataType AttentionLayer::output_type() const {
      return _linear.back().output_type();
    }

    dim_t AttentionLayer::output_size() const {
      return _d_model;
    }

    StorageView AttentionLayer::prepare_length_mask(const StorageView& lengths,
                                                        const dim_t num_heads,
                                                        const dim_t num_queries,
                                                        const bool mask_future,
                                                        const bool multi_query) {
      const Device device = lengths.device();
      const dim_t batch_size = lengths.size();
      StorageView mask(lengths.dtype(), device);

      if (multi_query)
        mask.resize({batch_size, num_queries, num_heads});
      else
        mask.resize({batch_size, num_heads, num_queries});

      DEVICE_DISPATCH(device, (primitives<D>::prepare_length_mask(lengths.data<int32_t>(),
                                                                  batch_size,
                                                                  num_heads,
                                                                  num_queries,
                                                                  mask_future,
                                                                  multi_query,
                                                                  mask.data<int32_t>())));
      return mask;
    }

  }
}
