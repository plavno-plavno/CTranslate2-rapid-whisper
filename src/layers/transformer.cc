#include "ctranslate2/layers/transformer.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

#ifdef CT2_WITH_CUDA
#  include <cuda_runtime.h>
#  include <spdlog/spdlog.h>
#  include "cuda/utils.h"
#  include "dispatch.h"
#  include "env.h"
#endif

namespace ctranslate2 {
  namespace layers {

    FeedForwardNetwork::FeedForwardNetwork(const models::Model& model,
                                           const std::string& scope,
                                           const bool pre_norm,
                                           const ops::ActivationType activation_type)
      : _layer_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _pre_norm(pre_norm)
      , _activation_type(activation_type)
      , _ff1(model, scope + "/linear_0", &_activation_type)
      , _ff1_noact(build_optional_layer<Dense>(model, scope + "/linear_0_noact"))
      , _ff2(model, scope + "/linear_1") {
    }

    void FeedForwardNetwork::operator()(const StorageView& input, StorageView& output) const {
      const StorageView* x = &input;
      if (_layer_norm && _pre_norm) {
        (*_layer_norm)(input, output);
        x = &output;
      }

      const Device device = input.device();
      const DataType dtype = input.dtype();

      StorageView inner(dtype, device);
      _ff1(*x, inner);
      if (_ff1_noact) {
        StorageView linear(dtype, device);
        (*_ff1_noact)(*x, linear);
        ops::Mul()(linear, inner, inner);
      }

      _ff2(inner, output, _layer_norm ? &input : nullptr);

      if (_layer_norm && !_pre_norm)
        (*_layer_norm)(output, output);
    }


    TransformerEncoderLayer::TransformerEncoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type)
      : _self_attention(std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm)))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/post_feedforward_layer_norm"))
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }


    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView* lengths,
                                             StorageView& output,
                                             const Padder* padder,
                                             StorageView* position_bias) const {
      PROFILE("TransformerEncoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      // Check if using pre_post_layer_norm pattern (T5Gemma style)
      const bool pre_post_layer_norm = _input_layer_norm && _post_attention_layer_norm
                                        && _pre_feedforward_layer_norm && _post_feedforward_layer_norm;

      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);

        (*_input_layer_norm)(input, hidden);

        if (_self_attention)
          (*_self_attention)(hidden,
                          hidden,
                          lengths,
                          context,
                          nullptr,
                          nullptr,
                          nullptr,
                          padder,
                          padder,
                          true,
                          position_bias);

        // post_self_attn_layernorm
        (*_post_attention_layer_norm)(context, output);

        // residual + hidden_states
        ops::Add()(input, output, output);

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        // mlp
        _ff(hidden, output);

        // post_feedforward_layernorm
        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);

        // residual + hidden_states
        ops::Add()(context, output, output);
        return;
      }

      // Original path for standard pre-norm/post-norm architectures
      StorageView context(dtype, device);
      if (_self_attention)
        (*_self_attention)(input,
                        input,
                        lengths,
                        context,
                        nullptr,
                        nullptr,
                        nullptr,
                        padder,
                        padder,
                        true,
                        position_bias);
      _ff(context, output);
    }


    TransformerDecoderLayer::TransformerDecoderLayer(const models::Model& model,
                                                     const std::string& scope,
                                                     const dim_t num_heads,
                                                     const bool pre_norm,
                                                     const ops::ActivationType activation_type)
      : _self_attention(std::unique_ptr<AttentionLayer>(new MultiHeadAttention(model,
                        scope + "/self_attention",
                        num_heads,
                        /*self_attention=*/true,
                        pre_norm,
                        /*is_decoder=*/true)))
      , _shared_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/shared_layer_norm"))
      , _input_layer_norm(build_optional_layer<LayerNorm>(model, scope + "/input_layer_norm"))
      , _post_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_attention_layer_norm"))
      , _pre_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/pre_feedforward_layer_norm"))
      , _post_feedforward_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/post_feedforward_layer_norm"))
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm,
                                                                    /*is_decoder=*/true))
      , _ff(model, scope + "/ffn", pre_norm, activation_type)
      , _external_pre_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_pre_encoder_attention_layer_norm"))
      , _external_post_encoder_attention_layer_norm(build_optional_layer<LayerNorm>(
                                     model, scope + "/external_post_encoder_attention_layer_norm"))
      , _layer_scalar(model.get_attribute_with_default<float>(scope + "/layer_scalar", 1.f))
      , _has_merged_encoder_attention(
          static_cast<MultiHeadAttention*>(_self_attention.get())->has_merged_encoder_attention())
      {
    }

    void TransformerDecoderLayer::operator()(const StorageView& input,
                                             const StorageView* input_length,
                                             const StorageView* memory,
                                             const StorageView* memory_lengths,
                                             StorageView* cached_self_attn_keys,
                                             StorageView* cached_self_attn_values,
                                             StorageView* cached_attn_keys,
                                             StorageView* cached_attn_values,
                                             StorageView& output,
                                             StorageView* attention,
                                             const Padder* input_padder,
                                             const Padder* memory_padder,
                                             bool return_normalized_attention,
                                             StorageView* position_bias,
                                             dim_t offset) const {
      PROFILE("TransformerDecoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      const bool pre_post_layer_norm = _post_feedforward_layer_norm && _pre_feedforward_layer_norm;
      if (pre_post_layer_norm) {
        StorageView hidden(dtype, device);
        StorageView context(dtype, device);
        (*_input_layer_norm)(input, hidden);

        if (_has_merged_encoder_attention) {
          static_cast<MultiHeadAttention*>(_self_attention.get())->forward_merged(
            hidden,
            memory,
            memory_lengths,
            input_length,
            context,
            cached_self_attn_keys,
            cached_self_attn_values,
            cached_attn_keys,
            cached_attn_values,
            input_padder,
            memory_padder,
            offset);
        } else if (_self_attention) {
          (*_self_attention)(hidden,
                             hidden,
                             input_length,
                             context,
                             cached_self_attn_keys,
                             cached_self_attn_values,
                             nullptr,
                             input_padder,
                             input_padder,
                             true,
                             position_bias,
                             offset);
        }
        (*_post_attention_layer_norm)(context, output);
        ops::Add()(output, input, output);

        if (_encoder_attention) {
            StorageView cross_attn_in = output;  // save for residual

            StorageView query_normalized(dtype, device);
            if (_external_pre_encoder_attention_layer_norm) {
                (*_external_pre_encoder_attention_layer_norm)(output, query_normalized);
            }
            else {
                query_normalized.shallow_copy(output);
            }

            (*_encoder_attention)(query_normalized,
                                  *memory,
                                  memory_lengths,
                                  context,
                                  cached_attn_keys,
                                  cached_attn_values,
                                  attention,
                                  input_padder,
                                  memory_padder,
                                  return_normalized_attention);

            if (_external_post_encoder_attention_layer_norm) {
                (*_external_post_encoder_attention_layer_norm)(context, context);
            }
            ops::Add()(context, cross_attn_in, output);
        }

        context = std::move(output);
        (*_pre_feedforward_layer_norm)(context, output);
        hidden = std::move(output);

        _ff(hidden, output);

        hidden = std::move(output);
        (*_post_feedforward_layer_norm)(hidden, output);
        ops::Add()(output, context, output);

        // Gemma 4 layer scalar
        if (_layer_scalar != 1.f)
          ops::Mul()(output, StorageView(_layer_scalar).to(dtype), output);

        return;
      }

      const bool use_parallel_residual = _shared_layer_norm || _input_layer_norm;

      if (use_parallel_residual) {
        // The parallel residual implementation assumes there is no cross attention.
        StorageView hidden(dtype, device);

        if (_shared_layer_norm)
          (*_shared_layer_norm)(input, hidden);
        else
          (*_input_layer_norm)(input, hidden);

        StorageView attn(dtype, device);
        if (_self_attention)
          (*_self_attention)(hidden,
                        hidden,
                        input_length,
                        attn,
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        nullptr,
                        input_padder,
                        input_padder,
                        true,
                        position_bias,
                        offset);

        if (_post_attention_layer_norm)
          (*_post_attention_layer_norm)(input, hidden);

        _ff(hidden, output);

        ops::Add()(output, input, output);
        ops::Add()(output, attn, output);

        return;
      }
      if (_self_attention)
        (*_self_attention)(input,
                      input,
                      input_length,
                      output,
                      cached_self_attn_keys,
                      cached_self_attn_values,
                      nullptr,
                      input_padder,
                      input_padder,
                      true,
                      position_bias,
                      offset);

      StorageView context(dtype, device);
      if (_encoder_attention) {
        (*_encoder_attention)(output,
                              *memory,
                              memory_lengths,
                              context,
                              cached_attn_keys,
                              cached_attn_values,
                              attention,
                              input_padder,
                              memory_padder,
                              return_normalized_attention);
      }
      else {
        context = std::move(output);
      }

      _ff(context, output);
    }


    static std::unique_ptr<PositionEncoder>
    build_position_encoder(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      if (model.get_variable_if_exists(scope + "/encodings"))
        return std::make_unique<PositionEmbedding>(model, scope);
      else
        return std::make_unique<SinusoidalPositionEncoder>(embeddings.output_size(),
                                                           embeddings.output_type(),
                                                           model.device());
    }

    static std::unique_ptr<const StorageView>
    build_embeddings_scale(const models::Model& model,
                           const std::string& scope,
                           const Layer& embeddings) {
      const auto* scale = model.get_variable_if_exists(scope + "/scale_embeddings");

      // Backward compatibility with older models.
      if (!scale)
        scale = model.get_variable_if_exists(scope + "/embeddings/multiply_by_sqrt_depth");

      StorageView value;

      // The attribute can either be a boolean flag or the actual scale value.
      if (!scale || (scale->dtype() == DataType::INT8 && scale->as_scalar<int8_t>()))
        value = StorageView(std::sqrt(static_cast<float>(embeddings.output_size())));
      else if (scale->dtype() != DataType::INT8 && scale->as_scalar<float>() != 1.f)
        value = *scale;
      else
        return nullptr;

      return std::make_unique<StorageView>(value.to(embeddings.output_type()));
    }


    // T5-style models compute the relative attention bias in the first layer and share it
    // with the following layers. UMT5-style models give every layer its own bias table.
    // Identical tables are serialized as aliases resolving to the same StorageView, so
    // comparing the resolved pointers tells the two cases apart.
    static bool has_shared_position_bias(const models::Model& model,
                                         const std::string& scope,
                                         const size_t num_layers) {
      const auto bias_name = [&scope](size_t i) {
        return scope + "/layer_" + std::to_string(i) + "/self_attention/relative_attention_bias";
      };

      const StorageView* first = model.get_variable_if_exists(bias_name(0));
      if (!first)
        return true;  // no relative attention bias: the shared buffer is never filled

      for (size_t i = 1; i < num_layers; ++i) {
        if (model.get_variable_if_exists(bias_name(i)) != first)
          return false;
      }

      return true;
    }


    TransformerEncoder::TransformerEncoder(const models::Model& model, const std::string& scope)
      : _embeddings(model, scope + "/embeddings",
                    model.get_enum_value<EmbeddingsMerge>(scope + "/embeddings_merge"))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _layers(build_layers_list<const TransformerEncoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation")))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _shared_position_bias(has_shared_position_bias(model, scope, _layers.size()))
    {
    }

    void TransformerEncoder::operator()(const std::vector<StorageView>& ids,
                                        const StorageView* lengths,
                                        StorageView& output) {
      PROFILE("TransformerEncoder");
      StorageView input(output.dtype(), output.device());
      _embeddings(ids, input);
      if (_embeddings_scale)
        ops::Mul()(input, *_embeddings_scale, input);
      if (_position_encoder)
        (*_position_encoder)(input);
      if (_layernorm_embedding)
        (*_layernorm_embedding)(input, input);

      const dim_t max_time = input.dim(1);

      // Remove padding to reduce the amount of computation.
      std::unique_ptr<Padder> padder;
      std::unique_ptr<StorageView> lengths_mask;

      if (lengths) {
        if (Padder::allow_padding_removal(output.device(), _compute_type)) {
          padder = std::make_unique<Padder>(*lengths, max_time);
          padder->remove_padding(input);
        }

        int num_heads = _num_heads;
        lengths_mask = std::make_unique<StorageView>(
          layers::MultiHeadAttention::prepare_length_mask(*lengths, num_heads, max_time));
      }

      StorageView position_bias(output.dtype(), output.device());
      StorageView* position_bias_ptr = _shared_position_bias ? &position_bias : nullptr;

      for (size_t l = 0; l < _layers.size(); ++l) {
        (*_layers[l])(input, lengths_mask.get(), output, padder.get(), position_bias_ptr);
        if (l + 1 < _layers.size())
          input = std::move(output);
      }
      if (_output_norm)
        (*_output_norm)(output, output);
      if (padder)
        padder->add_padding(output);
    }


    TransformerDecoder::TransformerDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _start_from_zero_embedding(model.get_flag_with_default(scope + "/start_from_zero_embedding",
                                                               false))
      , _embeddings_scale(build_embeddings_scale(model, scope, _embeddings))
      , _layernorm_embedding(build_optional_layer<LayerNorm>(model, scope + "/layernorm_embedding"))
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _project_in(build_optional_layer<Dense>(model, scope + "/project_in"))
      , _project_out(build_optional_layer<Dense>(model, scope + "/project_out"))
      , _layers(build_layers_list<const TransformerDecoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation")))
      , _position_encoder(_layers.front()->get_self_attention().has_positional_embeddings()
                          ? nullptr
                          : build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _proj(model, scope + "/projection")
      , _sliding_window(model.get_attribute_with_default<int32_t>(scope + "/sliding_window", 0))
      , _final_logit_softcapping(model.get_attribute_with_default<float>(scope + "/final_logit_softcapping", 0.f))
      , _shared_position_bias(has_shared_position_bias(model, scope, _layers.size())) {

      dim_t alignment_layer = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_layer", -1));
      dim_t alignment_heads = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_heads", 1));

      if (alignment_layer < 0)
        alignment_layer = _layers.size() + alignment_layer;
      if (alignment_heads == 0)
        alignment_heads = _num_heads;

      set_alignment_heads(alignment_layer, alignment_heads);

      const auto* outputs_scale = model.get_variable_if_exists(scope + "/scale_outputs");
      if (outputs_scale) {
        const DataType dtype = get_default_float_type(_compute_type);
        _outputs_scale = std::make_unique<StorageView>(outputs_scale->to(dtype));
      }
    }

    DecoderState TransformerDecoder::initial_state(bool iterative_decoding) const {
      DecoderState state;

      if (iterative_decoding) {
        const size_t state_size = _layers.size() * (_with_encoder_attention ? 4 : 2);
        state.reserve(state_size);

        const DataType dtype = output_type();

        for (size_t i = 0; i < _layers.size(); ++i) {
          const std::string i_str = std::to_string(i);
          state.emplace("self_keys_" + i_str, StorageView(dtype, _device));
          state.emplace("self_values_" + i_str, StorageView(dtype, _device));
          if (_with_encoder_attention) {
            state.emplace("memory_keys_" + i_str, StorageView(dtype, _device));
            state.emplace("memory_values_" + i_str, StorageView(dtype, _device));
          }
        }
      }

      return state;
    }

    bool TransformerDecoder::replicate_state(const std::string& name) const {
      // No need to replicate projected memory keys and values as they are the same for each beam.
      return !_with_encoder_attention || !starts_with(name, "memory");
    }

    void TransformerDecoder::set_alignment_heads(const dim_t layer,
                                                 const dim_t num_heads_to_average) {
      std::vector<dim_t> range(num_heads_to_average);
      std::iota(range.begin(), range.end(), dim_t(0));

      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      _alignment_heads[layer] = std::move(range);

      _average_alignment_heads = true;
    }

    void TransformerDecoder::set_alignment_heads(const std::vector<std::pair<dim_t, dim_t>>& alignment_heads) {
      _alignment_heads.clear();
      _alignment_heads.resize(_layers.size());
      for (const auto& [layer, head] : alignment_heads)
        _alignment_heads[layer].push_back(head);

      _average_alignment_heads = false;
    }

    std::unique_ptr<StorageView>
    TransformerDecoder::get_layer_alignment_heads(const dim_t layer, const dim_t batch_size) const {
      if (_alignment_heads.empty())
        return nullptr;

      const auto& heads = _alignment_heads[layer];
      const dim_t num_heads = heads.size();

      if (heads.empty())
        return nullptr;

      std::vector<int32_t> indices;
      indices.reserve(batch_size * num_heads);
      for (dim_t i = 0; i < batch_size; ++i)
        indices.insert(indices.end(), heads.begin(), heads.end());

      return std::make_unique<StorageView>(Shape{batch_size, num_heads}, indices, _device);
    }

#ifdef CT2_WITH_CUDA
    // A decoding step launches ~30 kernels per layer, each for a [batch, 1, depth] input,
    // so for small batches the step is bound by kernel launch overhead. With
    // CT2_CUDA_GRAPHS=1 each step is captured once into a CUDA graph and replayed in later
    // calls with the same step index (in streaming ASR every window decodes steps 1, 2, ...
    // again). The graph runs exactly the kernels of the regular path with the same shapes
    // and layouts, so the results are bit-identical; only the addresses are made stable:
    //  - the self-attention cache is concatenated into per-layer ping-pong buffers (step t
    //    reads buffer t % 2 and writes buffer (t + 1) % 2), with the same layout as the
    //    regular concatenation into a fresh tensor;
    //  - the cross-attention keys/values, the input ids and the logits live in persistent
    //    buffers; the state is copied into them when it is not already there (e.g. after
    //    the beam search reordered it), which is an exact device copy.
    // For greedy search the graph can also contain the token selection (queue_greedy_step):
    // the same token suppression, log softmax and top-1 ops as the search, the selected id
    // and score written to a ring of result slots, and the id mapped to the original
    // vocabulary straight into the input buffer of the next step. Steps can then be queued
    // without a host round trip.
    // Temporaries are allocated with cudaMallocAsync during the capture and become graph
    // memory nodes. Any capture failure disables the graphs and falls back to the regular
    // path. Graphs reference the model weights: they are dropped when the output layer
    // changes, and they must not outlive a model unload/reload.
    struct TransformerDecoder::StepGraphs {
      static constexpr dim_t num_result_slots = 64;

      struct Graph {
        cudaGraphExec_t exec = nullptr;
        std::vector<Shape> keys_shapes;
        std::vector<Shape> values_shapes;
        Shape logits_shape;
      };

      // (step, output size, selection: -1 none / 0 regular step / 1 begin step, options id)
      using GraphKey = std::tuple<dim_t, dim_t, int, size_t>;

      struct Buffers {
        std::vector<StorageView> keys[2];
        std::vector<StorageView> values[2];
        std::vector<StorageView> memory_keys;
        std::vector<StorageView> memory_values;
        std::vector<SelfAttentionCacheTarget> targets[2];
        StorageView ids;
        StorageView logits;
        dim_t batch_size = 0;
        dim_t max_logits_size = 0;
        // Step whose input ids are in `ids` after a queued greedy step (-1: none).
        dim_t ids_step = -1;
        // Greedy selection results, [num_result_slots][batch], and their host copies.
        StorageView result_ids;
        StorageView result_scores;
        int32_t* host_ids = nullptr;
        void* host_scores = nullptr;
        // (options id, begin step) -> flat indices of the suppressed logits
        std::map<std::pair<size_t, bool>, StorageView> disable_indices;
        std::map<GraphKey, Graph> graphs;

        ~Buffers() {
          for (auto& [key, graph] : graphs)
            if (graph.exec)
              cudaGraphExecDestroy(graph.exec);
          if (host_ids)
            cudaFreeHost(host_ids);
          if (host_scores)
            cudaFreeHost(host_scores);
        }
      };

      // (batch, heads, head depth, memory keys shape) -> buffers
      std::map<std::tuple<dim_t, dim_t, dim_t, Shape>, Buffers> buffers;
      std::vector<GreedyStepOptions> greedy_options;
      StorageView id_table;  // Output id -> original vocabulary id, on the device.
      Buffers* last_greedy_buffers = nullptr;
      dim_t output_layer_version = -1;
      bool disabled = false;

      void clear() {
        buffers.clear();
        greedy_options.clear();
        id_table = StorageView();
        last_greedy_buffers = nullptr;
      }
    };

    TransformerDecoder::~TransformerDecoder() = default;

    static bool cuda_graphs_enabled() {
      static const bool enabled = read_bool_from_env("CT2_CUDA_GRAPHS", false);
      return enabled;
    }

    static dim_t cuda_graphs_max_steps() {
      static const dim_t max_steps = read_int_from_env("CT2_CUDA_GRAPHS_MAX_STEPS", 448);
      return max_steps;
    }

    // Brings a state tensor into its persistent buffer (an exact device copy, skipped when
    // it already lives there) and makes the state entry a view of the buffer.
    static void stage_in_buffer(StorageView& entry, StorageView& buffer) {
      if (entry.buffer() == buffer.buffer() && entry.shape() == buffer.shape())
        return;
      if (entry.buffer() == buffer.buffer()) {
        const StorageView copy(entry);
        buffer.copy_from(copy);
      } else {
        buffer.copy_from(entry);
      }
      entry.shallow_copy(buffer);
    }

    dim_t TransformerDecoder::max_queued_greedy_steps() const {
      static const dim_t max_queued = std::max<dim_t>(
        1, std::min<dim_t>(read_int_from_env("CT2_CUDA_GRAPHS_QUEUE", 2),
                           StepGraphs::num_result_slots));
      static const bool device_greedy = read_bool_from_env("CT2_CUDA_GRAPHS_DEVICE_GREEDY", true);
      if (!cuda_graphs_enabled() || !device_greedy || _device != Device::CUDA
          || (_step_graphs && _step_graphs->disabled))
        return 0;
      return max_queued;
    }

    bool TransformerDecoder::queue_greedy_step(dim_t step,
                                               const GreedyStepOptions& options,
                                               const std::vector<int32_t>* input_ids,
                                               bool begin_step,
                                               DecoderState& state) {
      if (max_queued_greedy_steps() == 0 || step <= 0)
        return false;
      std::unique_ptr<StorageView> ids;
      if (input_ids)
        ids = std::make_unique<StorageView>(Shape{dim_t(input_ids->size())}, *input_ids, _device);
      return run_step_graph(step, ids.get(), state, nullptr, &options, begin_step);
    }

    void TransformerDecoder::read_greedy_steps(dim_t first_step,
                                               dim_t num_steps,
                                               std::vector<int32_t>& ids,
                                               std::vector<float>& scores) {
      StepGraphs::Buffers& buffers = *_step_graphs->last_greedy_buffers;
      const dim_t batch_size = buffers.batch_size;
      const dim_t item_size = buffers.result_scores.item_size();
      const auto* result_scores = static_cast<const char*>(buffers.result_scores.buffer());
      cudaStream_t stream = cuda::get_cuda_stream();

      for (dim_t k = 0; k < num_steps; ++k) {
        const dim_t slot = (first_step + k) % StepGraphs::num_result_slots;
        CUDA_CHECK(cudaMemcpyAsync(buffers.host_ids + k * batch_size,
                                   buffers.result_ids.data<int32_t>() + slot * batch_size,
                                   batch_size * sizeof (int32_t),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(buffers.host_scores)
                                   + k * batch_size * item_size,
                                   result_scores + slot * batch_size * item_size,
                                   batch_size * item_size,
                                   cudaMemcpyDeviceToHost,
                                   stream));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));

      const dim_t size = num_steps * batch_size;
      ids.assign(buffers.host_ids, buffers.host_ids + size);
      scores.resize(size);
      // Same conversion as StorageView::scalar_at<float> on the host scores.
      TYPE_DISPATCH(buffers.result_scores.dtype(),
                    for (dim_t i = 0; i < size; ++i)
                      scores[i] = static_cast<const T*>(buffers.host_scores)[i]);
    }

    bool TransformerDecoder::run_step_graph(dim_t step,
                                            const StorageView* ids,
                                            DecoderState& state,
                                            StorageView* logits,
                                            const GreedyStepOptions* greedy,
                                            bool begin_step) {
      if (!_step_graphs)
        _step_graphs = std::make_unique<StepGraphs>();
      StepGraphs& graphs = *_step_graphs;
      if (graphs.disabled)
        return false;
      if (_sliding_window > 0)
        return false;

      if (graphs.output_layer_version != output_layer_version()) {
        graphs.clear();
        graphs.output_layer_version = output_layer_version();
      }

      const dim_t num_layers = _layers.size();
      const StorageView& keys_0 = state.at("self_keys_0");
      if (keys_0.rank() != 4 || keys_0.dim(2) != step)
        return false;
      const dim_t batch_size = keys_0.dim(0);
      if (ids && ids->dim(0) != batch_size)
        return false;
      const dim_t num_heads = keys_0.dim(1);
      const dim_t depth = keys_0.dim(3);
      const Shape cache_shape = keys_0.shape();
      const dim_t max_cache_size = batch_size * num_heads * cuda_graphs_max_steps() * depth;
      if (keys_0.size() + batch_size * num_heads * depth > max_cache_size)
        return false;

      Shape memory_shape;
      if (_with_encoder_attention) {
        memory_shape = state.at("memory_keys_0").shape();
        if (memory_shape.empty())
          return false;
      }
      for (dim_t l = 0; l < num_layers; ++l) {
        const std::string l_str = std::to_string(l);
        if (state.at("self_keys_" + l_str).shape() != cache_shape
            || state.at("self_values_" + l_str).shape() != cache_shape)
          return false;
        if (_with_encoder_attention
            && (state.at("memory_keys_" + l_str).shape() != memory_shape
                || state.at("memory_values_" + l_str).shape() != memory_shape))
          return false;
      }

      const DataType dtype = output_type();
      const dim_t output_size = this->output_size();

      auto [buffers_it, inserted] = graphs.buffers.try_emplace(
        std::make_tuple(batch_size, num_heads, depth, memory_shape));
      StepGraphs::Buffers& buffers = buffers_it->second;
      if (inserted) {
        // Allocate all persistent buffers at their final capacity, once: their addresses
        // are baked into the graphs.
        dim_t memory_size = 1;
        for (const dim_t dim : memory_shape)
          memory_size *= dim;
        for (int parity = 0; parity < 2; ++parity) {
          for (dim_t l = 0; l < num_layers; ++l) {
            buffers.keys[parity].emplace_back(dtype, _device);
            buffers.keys[parity].back().reserve(max_cache_size);
            buffers.values[parity].emplace_back(dtype, _device);
            buffers.values[parity].back().reserve(max_cache_size);
          }
          for (dim_t l = 0; l < num_layers; ++l)
            buffers.targets[parity].push_back({&buffers.keys[parity][l],
                                               &buffers.values[parity][l]});
        }
        if (_with_encoder_attention) {
          for (dim_t l = 0; l < num_layers; ++l) {
            buffers.memory_keys.emplace_back(dtype, _device);
            buffers.memory_keys.back().reserve(memory_size);
            buffers.memory_values.emplace_back(dtype, _device);
            buffers.memory_values.back().reserve(memory_size);
          }
        }
        buffers.batch_size = batch_size;
        buffers.ids = StorageView(DataType::INT32, _device);
        buffers.ids.reserve(batch_size);
        buffers.max_logits_size = batch_size * output_size;
        buffers.logits = StorageView(dtype, _device);
        buffers.logits.reserve(buffers.max_logits_size);
        const dim_t num_results = StepGraphs::num_result_slots * batch_size;
        buffers.result_ids = StorageView({num_results}, DataType::INT32, _device);
        buffers.result_scores = StorageView({num_results}, dtype, _device);
        CUDA_CHECK(cudaMallocHost(&buffers.host_ids, num_results * sizeof (int32_t)));
        CUDA_CHECK(cudaMallocHost(&buffers.host_scores, num_results * sizeof (float)));
      }
      if (batch_size * output_size > buffers.max_logits_size)
        return false;
      if (!ids && buffers.ids_step != step)
        return false;  // The input ids of this step were not produced on the device.

      // Greedy selection: options id, device table of original ids, suppressed indices.
      size_t options_id = 0;
      const StorageView* disable_indices = nullptr;
      if (greedy) {
        const auto it = std::find(graphs.greedy_options.begin(),
                                  graphs.greedy_options.end(),
                                  *greedy);
        options_id = it - graphs.greedy_options.begin();
        if (it == graphs.greedy_options.end())
          graphs.greedy_options.push_back(*greedy);

        if (graphs.id_table.empty()) {
          std::vector<int32_t> table(output_size);
          for (dim_t i = 0; i < output_size; ++i)
            table[i] = to_original_word_id(i);
          graphs.id_table = StorageView({output_size}, table, _device);
        }

        auto [indices_it, new_indices] = buffers.disable_indices.try_emplace(
          std::make_pair(options_id, begin_step));
        if (new_indices) {
          // Same sorted, unique flat indices as DisableTokens.
          std::vector<int32_t> flat;
          for (dim_t b = 0; b < batch_size; ++b) {
            for (const size_t id : greedy->disable_ids)
              flat.push_back(b * output_size + id);
            if (begin_step)
              for (const size_t id : greedy->disable_ids_begin)
                flat.push_back(b * output_size + id);
          }
          std::sort(flat.begin(), flat.end());
          flat.erase(std::unique(flat.begin(), flat.end()), flat.end());
          if (!flat.empty())
            indices_it->second = StorageView({dim_t(flat.size())}, flat, _device);
        }
        disable_indices = &indices_it->second;
      }

      // Stage the inputs: the cache of length `step` must be in buffer step % 2.
      const int in = step % 2;
      const int out = 1 - in;
      for (dim_t l = 0; l < num_layers; ++l) {
        const std::string l_str = std::to_string(l);
        stage_in_buffer(state.at("self_keys_" + l_str), buffers.keys[in][l]);
        stage_in_buffer(state.at("self_values_" + l_str), buffers.values[in][l]);
        if (_with_encoder_attention) {
          stage_in_buffer(state.at("memory_keys_" + l_str), buffers.memory_keys[l]);
          stage_in_buffer(state.at("memory_values_" + l_str), buffers.memory_values[l]);
        }
      }
      if (ids) {
        buffers.ids.copy_from(*ids);
      } else {
        buffers.ids.reshape({batch_size});
      }

      cudaStream_t stream = cuda::get_cuda_stream();
      const StepGraphs::GraphKey graph_key(step,
                                           output_size,
                                           greedy ? int(begin_step) : -1,
                                           options_id);
      auto graph_it = buffers.graphs.find(graph_key);

      if (graph_it == buffers.graphs.end()) {
        // Capture. Nothing runs on the device here: decode() only records its kernels,
        // while its host side updates the state entries to views of the output buffers.
        _capture_cache_targets = &buffers.targets[out];
        std::string error;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
          decode(buffers.ids, nullptr, step, state, &buffers.logits, nullptr, true);
          if (greedy)
            select_greedy_on_device(buffers.logits,
                                    *greedy,
                                    *disable_indices,
                                    graphs.id_table,
                                    buffers.result_ids,
                                    buffers.result_scores,
                                    (step % StepGraphs::num_result_slots) * batch_size,
                                    buffers.ids);
        } catch (const std::exception& e) {
          error = e.what();
        }
        _capture_cache_targets = nullptr;
        self_attention_cache_target() = nullptr;

        cudaGraph_t graph = nullptr;
        cudaError_t status = cudaStreamEndCapture(stream, &graph);
        cudaGraphExec_t exec = nullptr;
        if (error.empty() && status == cudaSuccess)
          status = cudaGraphInstantiate(&exec, graph, 0);
        if (graph)
          cudaGraphDestroy(graph);

        if (!error.empty() || status != cudaSuccess) {
          cudaGetLastError();  // Clear the sticky capture error.
          if (error.empty())
            error = cudaGetErrorString(status);
          spdlog::warn("CUDA graph capture of the decoder step failed, disabling CUDA graphs: {}",
                       error);
          graphs.disabled = true;
          // Point the state back to the staged inputs, untouched by the capture.
          for (dim_t l = 0; l < num_layers; ++l) {
            const std::string l_str = std::to_string(l);
            state.at("self_keys_" + l_str).shallow_copy(buffers.keys[in][l]);
            state.at("self_values_" + l_str).shallow_copy(buffers.values[in][l]);
          }
          return false;
        }

        StepGraphs::Graph new_graph;
        new_graph.exec = exec;
        for (dim_t l = 0; l < num_layers; ++l) {
          new_graph.keys_shapes.push_back(buffers.keys[out][l].shape());
          new_graph.values_shapes.push_back(buffers.values[out][l].shape());
        }
        new_graph.logits_shape = buffers.logits.shape();
        graph_it = buffers.graphs.emplace(graph_key, std::move(new_graph)).first;

      } else {
        // Replay: reproduce the host side of the step.
        const StepGraphs::Graph& graph = graph_it->second;
        for (dim_t l = 0; l < num_layers; ++l) {
          const std::string l_str = std::to_string(l);
          buffers.keys[out][l].resize(graph.keys_shapes[l]);
          buffers.values[out][l].resize(graph.values_shapes[l]);
          state.at("self_keys_" + l_str).shallow_copy(buffers.keys[out][l]);
          state.at("self_values_" + l_str).shallow_copy(buffers.values[out][l]);
        }
        buffers.logits.resize(graph.logits_shape);
      }

      CUDA_CHECK(cudaGraphLaunch(graph_it->second.exec, stream));

      if (greedy) {
        buffers.ids.reshape({batch_size});
        buffers.ids_step = step + 1;
        graphs.last_greedy_buffers = &buffers;
      } else {
        buffers.ids_step = -1;
        logits->copy_from(buffers.logits);
      }
      return true;
    }

    // The token selection of GreedySearch with BestSampler, on the device: DisableTokens
    // (indexed_fill of the disable value), LogSoftMax when scores are returned, TopK(1).
    void TransformerDecoder::select_greedy_on_device(StorageView& logits,
                                                     const GreedyStepOptions& options,
                                                     const StorageView& disable_indices,
                                                     const StorageView& id_table,
                                                     StorageView& result_ids,
                                                     StorageView& result_scores,
                                                     dim_t result_offset,
                                                     StorageView& next_ids) const {
      const dim_t batch_size = logits.dim(0);

      if (!disable_indices.empty())
        TYPE_DISPATCH(logits.dtype(),
                      primitives<Device::CUDA>::indexed_fill(
                        logits.data<T>(),
                        static_cast<T>(options.disable_value),
                        disable_indices.data<int32_t>(),
                        disable_indices.size()));

      if (options.log_softmax)
        ops::LogSoftMax()(logits);

      StorageView best_ids({batch_size, 1}, result_ids.data<int32_t>() + result_offset, _device);
      StorageView best_scores(logits.dtype(), _device);
      TYPE_DISPATCH(logits.dtype(),
                    best_scores.view(result_scores.data<T>() + result_offset, {batch_size, 1}));
      ops::TopK(1)(logits, best_scores, best_ids);

      // Input of the next step: the selected ids in the original vocabulary.
      ops::Gather()(id_table, best_ids, next_ids);
    }
#endif

#ifndef CT2_WITH_CUDA
    struct TransformerDecoder::StepGraphs {};
    TransformerDecoder::~TransformerDecoder() = default;

    dim_t TransformerDecoder::max_queued_greedy_steps() const {
      return 0;
    }

    bool TransformerDecoder::queue_greedy_step(dim_t,
                                               const GreedyStepOptions&,
                                               const std::vector<int32_t>*,
                                               bool,
                                               DecoderState&) {
      return false;
    }

    void TransformerDecoder::read_greedy_steps(dim_t,
                                               dim_t,
                                               std::vector<int32_t>&,
                                               std::vector<float>&) {
    }
#endif

    void TransformerDecoder::operator()(dim_t step,
                                        const StorageView& ids,
                                        DecoderState& state,
                                        StorageView* logits,
                                        StorageView* attention) {
#ifdef CT2_WITH_CUDA
      if (logits && !attention && step > 0 && ids.rank() == 1 && _device == Device::CUDA
          && cuda_graphs_enabled() && run_step_graph(step, &ids, state, logits, nullptr, false))
        return;
#endif
      return decode(ids, nullptr, step, state, logits, attention);
    }

    void TransformerDecoder::operator()(const StorageView& ids,
                                        const StorageView& lengths,
                                        DecoderState& state,
                                        StorageView& logits,
                                        StorageView* attention) {
      return decode(ids, &lengths, -1, state, &logits, attention);
    }

    void TransformerDecoder::decode(const StorageView& ids,
                                    const StorageView* lengths,
                                    dim_t step,
                                    DecoderState& state,
                                    StorageView* outputs,
                                    StorageView* attention,
                                    bool return_logits) {
      PROFILE("TransformerDecoder");
      const DataType dtype = output_type();
      const Device device = ids.device();
      const bool is_sequence = ids.rank() > 1;

      StorageView layer_in(dtype, device);
      StorageView layer_out(dtype, device);

      _embeddings(ids, layer_in);
      if (_start_from_zero_embedding)
        zero_first_timestep(layer_in, step);
      if (_embeddings_scale && (!_start_from_zero_embedding || step != 0))
        ops::Mul()(layer_in, *_embeddings_scale, layer_in);
      if (_project_in) {
        (*_project_in)(layer_in, layer_out);
        layer_in = std::move(layer_out);
      }
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      if (_position_encoder)
        (*_position_encoder)(layer_in, std::max(step, dim_t(0)));
      if (_layernorm_embedding)
        (*_layernorm_embedding)(layer_in, layer_in);

      const dim_t batch_size = layer_in.dim(0);
      dim_t max_time;

      if (_sliding_window > 0 && layer_in.dim(1) > _sliding_window) {
        max_time = _sliding_window;
      } else
        max_time = layer_in.dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (is_sequence && !lengths) {
        input_lengths = std::make_unique<StorageView>(Shape{ids.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

      bool multi_query = _layers.front()->get_self_attention().multi_query();

      if (lengths) {
        if (allow_padding_removal) {
          input_padder = std::make_unique<Padder>(*lengths, max_time);
          input_padder->remove_padding(layer_in);
        }

        dim_t num_heads = _num_heads;

        StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
          *lengths,
          num_heads,
          max_time,
          /*mask_future=*/true,
          multi_query);


        if (step > 0)
          ops::Add()(lengths_mask, StorageView(int32_t(step)), lengths_mask);

        input_lengths_mask = std::make_unique<StorageView>(std::move(lengths_mask));
      }

      StorageView* memory = nullptr;
      std::unique_ptr<const StorageView> memory_lengths_mask;
      std::unique_ptr<const Padder> memory_padder;
      if (_with_encoder_attention) {
        const auto it = state.find("memory_lengths");
        const StorageView* memory_lengths = it != state.end() ? &it->second : nullptr;

        if (step <= 0) {
          memory = &state.at("memory");

          if (memory_lengths && allow_padding_removal) {
            memory_padder = std::make_unique<Padder>(*memory_lengths, memory->dim(1));
            memory_padder->remove_padding(*memory);
          }
        }

        if (memory_lengths) {
          dim_t num_heads = _num_heads;
          const dim_t beam_size = batch_size / memory_lengths->dim(0);
          memory_lengths_mask = std::make_unique<StorageView>(
            layers::MultiHeadAttention::prepare_length_mask(*memory_lengths,
                                                            num_heads,
                                                            beam_size > 1 ? beam_size : max_time));
        }
      }

      std::vector<StorageView> alignment_heads;
      if (attention)
        alignment_heads.reserve(_layers.size());

      StorageView position_bias(dtype, device);
      StorageView* position_bias_ptr = _shared_position_bias ? &position_bias : nullptr;

      std::vector<StorageView> layer_ins;

      while (true) {
        dim_t prompt_size = layer_in.dim(1);
        if (_sliding_window == 0 || prompt_size <= _sliding_window) {
          layer_ins.push_back(std::move(layer_in));
          break;
        }
        if (layer_in.dim(1) > _sliding_window) {
          StorageView tmp(dtype, device);
          const ops::Split split_op(1, {_sliding_window, prompt_size - _sliding_window});
          split_op(layer_in, tmp, layer_in);
          layer_ins.push_back(std::move(tmp));
        }
      }

      for (size_t i = 0; i < layer_ins.size(); ++i) {
        StorageView* layer_in_chunk = &layer_ins[i];
        for (size_t l = 0; l < _layers.size(); ++l) {
          StorageView* cached_self_attn_keys = nullptr;
          StorageView* cached_self_attn_values = nullptr;
          StorageView* cached_attn_keys = nullptr;
          StorageView* cached_attn_values = nullptr;

          if (step >= 0) {
            const std::string l_str = std::to_string(l);
            cached_self_attn_keys = &state.at("self_keys_" + l_str);
            cached_self_attn_values = &state.at("self_values_" + l_str);
            if (_with_encoder_attention) {
              cached_attn_keys = &state.at("memory_keys_" + l_str);
              cached_attn_values = &state.at("memory_values_" + l_str);
            }
          }

          std::unique_ptr<StorageView> heads_to_select = get_layer_alignment_heads(l, batch_size);
          std::unique_ptr<StorageView> layer_attention;
          if (attention && heads_to_select)
            layer_attention = std::make_unique<StorageView>(dtype, device);

          dim_t offset = _sliding_window * i + step;
          offset = offset < 0 ? 0 : offset;
          if (i > 0) {
            auto max_tokens = _sliding_window + layer_in_chunk->dim(1);
            StorageView tmp_lengths = StorageView(Shape{layer_in_chunk->dim(0)}, int32_t(max_tokens), device);
            int num_heads = _num_heads;
            StorageView lengths_mask = layers::MultiHeadAttention::prepare_length_mask(
              tmp_lengths,
              num_heads,
              max_tokens,
              /*mask_future=*/true,
              multi_query);

            const ops::Slide slide_lengths_op(2, _sliding_window, layer_in_chunk->dim(1));
            // reuse tmp_lengths
            slide_lengths_op(lengths_mask, tmp_lengths);
            input_lengths_mask = std::make_unique<StorageView>(std::move(tmp_lengths));
          }

          if (_capture_cache_targets)
            self_attention_cache_target() = &(*_capture_cache_targets)[l];
          (*_layers[l])(*layer_in_chunk,
                        input_lengths_mask.get(),
                        memory,
                        memory_lengths_mask.get(),
                        cached_self_attn_keys,
                        cached_self_attn_values,
                        cached_attn_keys,
                        cached_attn_values,
                        layer_out,
                        layer_attention.get(),
                        input_padder.get(),
                        memory_padder.get(),
                        return_normalized_attention(),
                        position_bias_ptr,
                        offset);
          self_attention_cache_target() = nullptr;
          *layer_in_chunk = std::move(layer_out);

          if (layer_attention) {
            alignment_heads.emplace_back(dtype, device);
            ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
          }
        }
        layer_in = std::move(*layer_in_chunk);
      }

      if (step == 0) {
        // The memory is no longer needed as its projections were cached in the first step.
        state.erase("memory");
      }

      if (attention && !alignment_heads.empty()) {
        if (_average_alignment_heads) {
          ops::Mean(1)(alignment_heads[0], *attention);
          if (!is_sequence)
            attention->squeeze(1);

        } else {
          std::vector<const StorageView*> alignment_heads_ptr;
          alignment_heads_ptr.reserve(alignment_heads.size());
          for (const auto& heads : alignment_heads)
            alignment_heads_ptr.emplace_back(&heads);

          ops::Concat(1)(alignment_heads_ptr, *attention);
          if (!is_sequence)
            attention->squeeze(2);
        }
      }

      if (outputs) {
        if (_output_norm)
          (*_output_norm)(layer_in, layer_in);
        if (_project_out) {
          (*_project_out)(layer_in, layer_out);
          layer_in = std::move(layer_out);
        }

        if (_outputs_scale)
          ops::Mul()(layer_in, *_outputs_scale, layer_in);

        if (return_logits) {
          _proj(layer_in, *outputs);
          if (_final_logit_softcapping != 0.f) {
            // logits = tanh(logits / cap) * cap  — squashes logits to (-cap, cap)
            const auto dtype = outputs->dtype();
            const auto device = outputs->device();
            ops::Mul()(*outputs, StorageView(1.f / _final_logit_softcapping).to(dtype), *outputs);
            ops::Tanh()(*outputs, *outputs);
            ops::Mul()(*outputs, StorageView(_final_logit_softcapping).to(dtype), *outputs);
          }
        } else
          *outputs = std::move(layer_in);

        if (!is_sequence)
          outputs->squeeze(1);
        else if (input_padder)
          input_padder->add_padding(*outputs);
      }
    }

  }
}
