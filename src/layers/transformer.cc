#include "ctranslate2/layers/transformer.h"

#include <algorithm>
#include <cmath>
#include <list>
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
      , _ff(model, scope + "/ffn", pre_norm, activation_type) {
    }


    void TransformerEncoderLayer::operator()(const StorageView& input,
                                             const StorageView* lengths,
                                             StorageView& output,
                                             const Padder* padder) const {
      PROFILE("TransformerEncoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      StorageView context(dtype, device);
      (*_self_attention)(input,
                         input,
                         lengths,
                         context,
                         nullptr,
                         nullptr,
                         nullptr,
                         padder,
                         padder,
                         true);
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
      , _encoder_attention(build_optional_layer<MultiHeadAttention>(model,
                                                                    scope + "/attention",
                                                                    num_heads,
                                                                    /*self_attention=*/false,
                                                                    pre_norm,
                                                                    /*is_decoder=*/true))
      , _ff(model, scope + "/ffn", pre_norm, activation_type)
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
                                             bool return_normalized_attention) const {
      PROFILE("TransformerDecoderLayer");

      const DataType dtype = input.dtype();
      const Device device = input.device();

      (*_self_attention)(input,
                         input,
                         input_length,
                         output,
                         cached_self_attn_keys,
                         cached_self_attn_values,
                         nullptr,
                         input_padder,
                         input_padder,
                         true);

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
      (void)embeddings;
      if (!model.get_variable_if_exists(scope + "/encodings"))
        throw std::invalid_argument("Position encodings " + scope + "/encodings are missing "
                                    "(sinusoidal encoders are not supported by this build)");
      return std::make_unique<PositionEmbedding>(model, scope);
    }

    TransformerDecoder::TransformerDecoder(const models::Model& model, const std::string& scope)
      : Decoder(model.device())
      , _num_heads(model.get_attribute_with_default<int32_t>(scope + "/num_heads", 8))
      , _compute_type(model.effective_compute_type())
      , _embeddings(model, scope + "/embeddings")
      , _output_norm(build_optional_layer<LayerNorm>(model, scope + "/layer_norm"))
      , _layers(build_layers_list<const TransformerDecoderLayer>(
                  model,
                  scope + "/layer",
                  _num_heads,
                  model.get_flag_with_default(scope + "/pre_norm", true),
                  model.get_enum_value<ops::ActivationType>(scope + "/activation")))
      , _position_encoder(build_position_encoder(model, scope + "/position_encodings", _embeddings))
      , _with_encoder_attention(_layers.front()->has_cross_attention())
      , _proj(model, scope + "/projection") {

      // Whisper models save these decoder options, always disabled.
      for (const char* flag : {"/alibi", "/start_from_zero_embedding"}) {
        if (model.get_flag_with_default(scope + flag, false))
          throw std::invalid_argument("Unsupported decoder option " + scope + flag
                                      + " in this Whisper-only build");
      }
      for (const char* option : {"/sliding_window", "/final_logit_softcapping", "/scale_outputs",
                                 "/project_in/weight", "/project_out/weight",
                                 "/layernorm_embedding/gamma"}) {
        if (model.get_variable_if_exists(scope + option))
          throw std::invalid_argument("Unsupported decoder option " + scope + option
                                      + " in this Whisper-only build");
      }
      const auto* scale = model.get_variable_if_exists(scope + "/scale_embeddings");
      if (!scale || (scale->dtype() == DataType::INT8 ? scale->as_scalar<int8_t>() != 0
                                                       : scale->as_scalar<float>() != 1.f))
        throw std::invalid_argument("Scaled decoder embeddings are not supported by this build");

      dim_t alignment_layer = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_layer", -1));
      dim_t alignment_heads = (
        model.get_attribute_with_default<int32_t>(scope + "/alignment_heads", 1));

      if (alignment_layer < 0)
        alignment_layer = _layers.size() + alignment_layer;
      if (alignment_heads == 0)
        alignment_heads = _num_heads;

      set_alignment_heads(alignment_layer, alignment_heads);
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

      // (batch, heads, head depth, memory keys shape) -> buffers. A set costs the
      // self-attention caches at full capacity plus the cross-attention keys/values
      // (0.4 GB for large-v3 batch 1, ~2 GB at batch 5), so only the most recently used
      // sets are kept (CT2_CUDA_GRAPHS_MAX_BUFFER_SETS, default 2).
      using BufferKey = std::tuple<dim_t, dim_t, dim_t, Shape>;
      std::map<BufferKey, Buffers> buffers;
      std::list<BufferKey> recently_used;
      std::vector<GreedyStepOptions> greedy_options;
      StorageView id_table;  // Output id -> original vocabulary id, on the device.
      Buffers* last_greedy_buffers = nullptr;
      dim_t output_layer_version = -1;
      bool disabled = false;

      void touch(const BufferKey& key) {
        recently_used.remove(key);
        recently_used.push_front(key);
      }

      // Frees the least recently used sets beyond max_sets, never `keep`. The state no
      // longer views them: it was staged into `keep` before this is called.
      void evict(const BufferKey& keep, size_t max_sets) {
        while (buffers.size() > max_sets && recently_used.size() > 1) {
          const BufferKey victim = recently_used.back();
          if (victim == keep)
            break;
          // Pending work may still read the victim's buffers (state staging copies).
          cudaStreamSynchronize(cuda::get_cuda_stream());
          auto it = buffers.find(victim);
          if (last_greedy_buffers == &it->second)
            last_greedy_buffers = nullptr;
          buffers.erase(it);
          recently_used.pop_back();
        }
      }

      void clear() {
        buffers.clear();
        recently_used.clear();
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
          || !ScopedStepGraphs::active() || (_step_graphs && _step_graphs->disabled))
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

      const StepGraphs::BufferKey buffers_key(batch_size, num_heads, depth, memory_shape);
      auto [buffers_it, inserted] = graphs.buffers.try_emplace(buffers_key);
      StepGraphs::Buffers& buffers = buffers_it->second;
      graphs.touch(buffers_key);
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
      static const size_t max_buffer_sets = std::max(
        1, read_int_from_env("CT2_CUDA_GRAPHS_MAX_BUFFER_SETS", 2));
      graphs.evict(buffers_key, max_buffer_sets);

      cudaStream_t stream = cuda::get_cuda_stream();
      const StepGraphs::GraphKey graph_key(step,
                                           output_size,
                                           greedy ? int(begin_step) : -1,
                                           options_id);
      auto graph_it = buffers.graphs.find(graph_key);

      if (graph_it == buffers.graphs.end()) {
        // Capture. Nothing runs on the device here: decode() only records its kernels,
        // while its host side updates the state entries to views of the output buffers.
        // The capture runs on a non-blocking stream of this thread (see
        // cuda::get_capture_stream), the graph is then launched on the regular stream.
        _capture_cache_targets = &buffers.targets[out];
        std::string error;
        cudaGraph_t graph = nullptr;
        cudaError_t status = cudaSuccess;
        {
          const cudaStream_t capture_stream = cuda::get_capture_stream();
          const cuda::ScopedCudaStream scoped_stream(capture_stream);
          CUDA_CHECK(cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal));
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

          status = cudaStreamEndCapture(capture_stream, &graph);
        }
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
          && ScopedStepGraphs::active()
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
      if (layer_in.rank() == 2)
        layer_in.expand_dims(1);
      (*_position_encoder)(layer_in, std::max(step, dim_t(0)));

      const dim_t batch_size = layer_in.dim(0);
      const dim_t max_time = layer_in.dim(1);

      const bool allow_padding_removal = Padder::allow_padding_removal(_device, _compute_type);

      std::unique_ptr<const Padder> input_padder;
      std::unique_ptr<const StorageView> input_lengths;
      std::unique_ptr<const StorageView> input_lengths_mask;

      if (is_sequence && !lengths) {
        input_lengths = std::make_unique<StorageView>(Shape{ids.dim(0)}, int32_t(max_time), device);
        lengths = input_lengths.get();
      }

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
          /*mask_future=*/true);

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

        if (_capture_cache_targets)
          self_attention_cache_target() = &(*_capture_cache_targets)[l];
        (*_layers[l])(layer_in,
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
                      return_normalized_attention());
        self_attention_cache_target() = nullptr;
        layer_in = std::move(layer_out);

        if (layer_attention) {
          alignment_heads.emplace_back(dtype, device);
          ops::Gather(1, 1)(*layer_attention, *heads_to_select, alignment_heads.back());
        }
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

        if (return_logits) {
          _proj(layer_in, *outputs);
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
