#include "ctranslate2/layers/common.h"

#include <cmath>

#include "ctranslate2/ops/activation.h"
#include "dispatch.h"

namespace ctranslate2 {
  namespace layers {

    StorageView
    make_sequence_inputs(const std::vector<std::vector<size_t>>& ids,
                         const Device device,
                         const dim_t length_multiple_of,
                         StorageView* lengths) {
      const dim_t batch_size = ids.size();

      if (lengths)
        *lengths = StorageView({batch_size}, DataType::INT32);

      // Record lengths and maximum length.
      dim_t max_length = 0;
      for (dim_t i = 0; i < batch_size; ++i) {
        const dim_t length = ids[i].size();
        if (lengths)
          lengths->at<int32_t>(i) = length;
        max_length = std::max(max_length, length);
      }

      if (max_length % length_multiple_of != 0) {
        max_length += (length_multiple_of - max_length % length_multiple_of);
      }

      // Make 2D input.
      StorageView input({batch_size, max_length}, int32_t(0));
      for (dim_t i = 0; i < batch_size; ++i) {
        const dim_t length = ids[i].size();
        for (dim_t t = 0; t < length; ++t)
          input.at<int32_t>({i, t}) = ids[i][t];
      }

      if (lengths)
        *lengths = lengths->to(device);
      return input.to(device);
    }


    Embeddings::Embeddings(const models::Model& model, const std::string& scope)
      : _embeddings(model.get_variable(scope + "/weight"))
      , _output_type(get_default_float_type(model.effective_compute_type()))
    {
    }

    DataType Embeddings::output_type() const {
      return _output_type;
    }

    dim_t Embeddings::output_size() const {
      return _embeddings.dim(1);
    }

    void Embeddings::operator()(const StorageView& ids,
                                StorageView& output) const {
      PROFILE("Embeddings");
      _gather_op(_embeddings, ids, output);
    }


    void PositionEncoder::operator()(StorageView& input, dim_t index) {
      const dim_t time = input.dim(1);
      const dim_t depth = input.dim(-1);
      const dim_t max_time = time + index;
      const StorageView& encodings = get_position_encoding(max_time);
      const dim_t num_encodings = encodings.dim(0);

      if (max_time > num_encodings)
        throw std::runtime_error("No position encodings are defined for positions >= "
                                 + std::to_string(num_encodings)
                                 + ", but got position "
                                 + std::to_string(max_time - 1));
      if (depth != encodings.dim(1))
        throw std::invalid_argument("Shape mismatch: position encodings have depth "
                                    + std::to_string(encodings.dim(1))
                                    + ", but the input has depth "
                                    + std::to_string(depth));

      DEVICE_AND_TYPE_DISPATCH(input.device(), input.dtype(),
                               primitives<D>::add_batch_broadcast(encodings.data<T>() + index * depth,
                                                                  input.data<T>(),
                                                                  time * depth,
                                                                  input.size()));
    }

    void PositionEncoder::operator()(const StorageView& input, StorageView& output, dim_t index) {
      output = input;
      operator()(output, index);
    }


    PositionEmbedding::PositionEmbedding(const models::Model& model, const std::string& scope)
      : _encoding(model.get_variable(scope + "/encodings"))
    {
    }

    const StorageView& PositionEmbedding::get_position_encoding(dim_t) {
      return _encoding;
    }

    DataType PositionEmbedding::output_type() const {
      return _encoding.dtype();
    }

    dim_t PositionEmbedding::output_size() const {
      return _encoding.dim(1);
    }

    dim_t PositionEmbedding::num_positions() const {
      return _encoding.dim(0);
    }


    Dense::Dense(const models::Model& model,
                 const std::string& scope,
                 const ops::ActivationType* activation_type)
      : _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _partial_weight(_weight.device(), _weight.dtype())
      , _partial_bias(_weight.device(), _bias ? _bias->dtype() : DataType::FLOAT32)
      , _output_type(get_default_float_type(model.effective_compute_type()))
      , _gemm_op(/*alpha=*/1,
                 /*beta=*/0,
                 /*trans_a=*/false,
                 /*trans_b=*/true,
                 /*a_is_packed=*/false,
                 /*b_is_packed=*/false,
                 activation_type)
    {
    }

    DataType Dense::output_type() const {
      return _output_type;
    }

    dim_t Dense::output_size() const {
      return _partial_weight ? _partial_weight.dim(0) : _weight.dim(0);
    }

    void Dense::select_weights(const StorageView* index, const StorageView* extra_bias) {
      if (index) {
        ops::Gather()(_weight, *index, _partial_weight);

        if (_bias) {
          ops::Gather()(*_bias, *index, _partial_bias);
          if (extra_bias)
            ops::Add()(_partial_bias, *extra_bias, _partial_bias);
        } else if (extra_bias) {
          _partial_bias = *extra_bias;
        }
      } else {
        _partial_weight.clear();
        _partial_bias.clear();
      }
    }

    void Dense::operator()(const StorageView& input, StorageView& output, const StorageView* residual) const {
      PROFILE("Dense");
      const StorageView* weight = _partial_weight.empty() ? &_weight : &_partial_weight;
      const StorageView* bias = _partial_bias.empty() ? _bias : &_partial_bias;
      _gemm_op(input, *weight, output, nullptr, bias, residual);
    }


    LayerNorm::LayerNorm(const models::Model& model, const std::string& scope)
      : _beta(model.get_variable_if_exists(scope + "/beta"))
      , _gamma(model.get_variable(scope + "/gamma"))
    {
      auto epsilon_it = model.config.find("layer_norm_epsilon");
      if (epsilon_it == model.config.end() || epsilon_it->is_null())
        _epsilon = _beta ? 1e-5 : 1e-6;
      else
        _epsilon = epsilon_it->get<float>();
    }

    DataType LayerNorm::output_type() const {
      return _gamma.dtype();
    }

    dim_t LayerNorm::output_size() const {
      return _gamma.size();
    }

    void LayerNorm::operator()(const StorageView& input, StorageView& output) const {
      if (!_beta)
        throw std::runtime_error("LayerNorm without beta (RMSNorm) is not supported by this build");
      const ops::LayerNorm norm_op(-1, _epsilon);
      norm_op(*_beta, _gamma, input, output);
    }


    Conv1D::Conv1D(const models::Model& model,
                   const std::string& scope,
                   dim_t stride,
                   dim_t padding,
                   dim_t dilation,
                   dim_t groups,
                   const ops::ActivationType* activation_type)
      : _conv_op(stride, padding, dilation, groups, activation_type)
      , _weight(model.get_variable(scope + "/weight"))
      , _bias(model.get_variable_if_exists(scope + "/bias"))
      , _qscale(model.get_variable_if_exists(scope + "/weight_scale")) {
    }

    DataType Conv1D::output_type() const {
      return _weight.dtype();
    }

    dim_t Conv1D::output_size() const {
      return _weight.dim(0);
    }

    dim_t Conv1D::input_size() const {
      return _weight.dim(1);
    }

    void Conv1D::operator()(const StorageView& input, StorageView& output) const {
      if (_bias)
        _conv_op(input, _weight, *_bias, output, _qscale);
      else
        _conv_op(input, _weight, output, _qscale);
    }

  }
}
