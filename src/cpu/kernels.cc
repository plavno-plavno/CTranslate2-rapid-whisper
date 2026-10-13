#include "cpu/kernels.h"

#include <limits>

#define TARGET_ISA CpuIsa::GENERIC
#include "cpu/vec.h"

// Associative arithmetic is required for the compiler to automatically vectorize loops.
// With GCC this can be minimally enabled with -fassociative-math but this flag is not
// available in other compiler.
//
// The macros below can be used to enabled fast-math locally.
//
// See https://github.com/marian-nmt/marian-dev/blob/1.11.0/src/common/definitions.h#L25-L78
// for more details.
#if defined(_MSC_VER)
#  define CT2_FFAST_MATH_BEGIN __pragma(float_control(precise, off, push))
#  define CT2_FFAST_MATH_END __pragma(float_control(pop))
#elif defined(__clang__)
#  define CT2_FFAST_MATH_BEGIN _Pragma("float_control(precise, off, push)")
#  define CT2_FFAST_MATH_END _Pragma("float_control(pop)")
#elif defined(__GNUC__)
#  define CT2_FFAST_MATH_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"-ffast-math\")")
#  define CT2_FFAST_MATH_END _Pragma("GCC pop_options")
#endif

#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {
  namespace cpu {

    template <CpuIsa ISA, typename T, typename Function>
    static void vectorized_unary_transform(const T* x, T* y, dim_t size, const Function& func) {
      const dim_t remaining = size % Vec<T, ISA>::width;
      size -= remaining;

      for (dim_t i = 0; i < size; i += Vec<T, ISA>::width) {
        auto v = Vec<T, ISA>::load(x + i);
        Vec<T, ISA>::store(func(v), y + i);
      }

      if (remaining != 0) {
        auto v = Vec<T, ISA>::load(x + size, remaining);
        Vec<T, ISA>::store(func(v), y + size, remaining);
      }
    }

    template <CpuIsa ISA, typename T, typename Function>
    static void vectorized_binary_transform(const T* a,
                                            const T* b,
                                            T* c,
                                            dim_t size,
                                            const Function& func) {
      const dim_t remaining = size % Vec<T, ISA>::width;
      size -= remaining;

      for (dim_t i = 0; i < size; i += Vec<T, ISA>::width) {
        auto v1 = Vec<T, ISA>::load(a + i);
        auto v2 = Vec<T, ISA>::load(b + i);
        Vec<T, ISA>::store(func(v1, v2), c + i);
      }

      if (remaining != 0) {
        auto v1 = Vec<T, ISA>::load(a + size, remaining);
        auto v2 = Vec<T, ISA>::load(b + size, remaining);
        Vec<T, ISA>::store(func(v1, v2), c + size, remaining);
      }
    }

    template <CpuIsa ISA,
              typename T,
              typename VecMapFunc,
              typename VecReduceFunc,
              typename VecHorizontalReduceFunc,
              typename ScalarMapFunc,
              typename ScalarReduceFunc>
    static T vectorized_map_reduce_all(const T* x,
                                       dim_t size,
                                       T init,
                                       const VecMapFunc& vec_map_func,
                                       const VecReduceFunc& vec_reduce_func,
                                       const VecHorizontalReduceFunc& vec_horizontal_reduce_func,
                                       const ScalarMapFunc& scalar_map_func,
                                       const ScalarReduceFunc& scalar_reduce_func) {
      if (Vec<T, ISA>::width == 1 || size <= Vec<T, ISA>::width) {
        T accu = init;
        for (dim_t i = 0; i < size; ++i) {
          accu = scalar_reduce_func(accu, scalar_map_func(x[i]));
        }
        return accu;
      }

      const dim_t remaining = size % Vec<T, ISA>::width;
      size -= remaining;

      auto vec_accu = Vec<T, ISA>::load(init);
      for (dim_t i = 0; i < size; i += Vec<T, ISA>::width) {
        auto v = Vec<T, ISA>::load(x + i);
        vec_accu = vec_reduce_func(vec_accu, vec_map_func(v));
      }

      auto accu = vec_horizontal_reduce_func(vec_accu);
      if (remaining != 0) {
        for (dim_t i = size; i < size + remaining; ++i)
          accu = scalar_reduce_func(accu, scalar_map_func(x[i]));
      }

      return accu;
    }

    template <CpuIsa ISA,
              typename T,
              typename VecReduceFunc,
              typename VecHorizontalReduceFunc,
              typename ScalarReduceFunc>
    static T vectorized_reduce_all(const T* x,
                                   dim_t size,
                                   T init,
                                   const VecReduceFunc& vec_reduce_func,
                                   const VecHorizontalReduceFunc& vec_horizontal_reduce_func,
                                   const ScalarReduceFunc& scalar_reduce_func) {
      return vectorized_map_reduce_all<ISA>(x,
                                            size,
                                            init,
                                            identity(),
                                            vec_reduce_func,
                                            vec_horizontal_reduce_func,
                                            identity(),
                                            scalar_reduce_func);
    }

    struct relu_func {
      vec_type<float, TARGET_ISA> operator()(vec_type<float, TARGET_ISA> v) const {
        using VecType = Vec<float, TARGET_ISA>;
        return VecType::max(v, VecType::load(0.f));
      }
    };

    struct gelu_func {
      vec_type<float, TARGET_ISA> operator()(vec_type<float, TARGET_ISA> v) const {
        using VecType = Vec<float, TARGET_ISA>;
        auto u = VecType::mul(VecType::load(0.7071067811865475f), v);
        u = VecType::add(VecType::load(1.f), VecType::erf(u));
        u = VecType::mul(v, u);
        u = VecType::mul(VecType::load(0.5f), u);
        return u;
      }
    };

    struct gelu_tanh_func {
      vec_type<float, TARGET_ISA> operator()(vec_type<float, TARGET_ISA> v) const {
        using VecType = Vec<float, TARGET_ISA>;
        auto u = VecType::mul(VecType::mul(v, v), v);
        u = VecType::mul_add(VecType::load(0.044715f), u, v);
        u = VecType::mul(VecType::load(0.7978845608028654f), u);
        u = VecType::tanh(u);
        u = VecType::add(VecType::load(1.f), u);
        u = VecType::mul(v, u);
        u = VecType::mul(VecType::load(0.5f), u);
        return u;
      }
    };

    struct gelu_sigmoid_func {
      vec_type<float, TARGET_ISA> operator()(vec_type<float, TARGET_ISA> v) const {
        using VecType = Vec<float, TARGET_ISA>;
        return VecType::div(v, VecType::add(VecType::load(1.f),
                                            VecType::exp(VecType::mul(VecType::load(-1.702f), v))));
      }
    };

    struct sigmoid_func {
      vec_type<float, TARGET_ISA> operator()(vec_type<float, TARGET_ISA> v) const {
        using VecType = Vec<float, TARGET_ISA>;
        return VecType::div(VecType::load(1.f), VecType::add(VecType::load(1.f), VecType::exp(VecType::neg(v))));
      }
    };

    struct swish_func {
      vec_type<float, TARGET_ISA> operator()(vec_type<float, TARGET_ISA> v) const {
        using VecType = Vec<float, TARGET_ISA>;
        return VecType::div(v, VecType::add(VecType::load(1.f), VecType::exp(VecType::neg(v))));
      }
    };

    struct tanh_func {
      vec_type<float, TARGET_ISA> operator()(vec_type<float, TARGET_ISA> v) const {
        using VecType = Vec<float, TARGET_ISA>;
        return VecType::tanh(v);
      }
    };


    template <CpuIsa ISA, typename T>
    void rcp(const T* x, T* y, dim_t size) {
      vectorized_unary_transform<ISA>(x, y, size, Vec<T, ISA>::rcp);
    }

    template<>
    void exp<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::exp);
    }

    template<>
    void log<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::log);
    }

    template<>
    void sin<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::sin);
    }

    template<>
    void cos<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::cos);
    }

    template<>
    void tanh<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, Vec<float, TARGET_ISA>::tanh);
    }

    template<>
    void gelu<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, gelu_func());
    }

    template<>
    void gelu_tanh<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, gelu_tanh_func());
    }

    template<>
    void gelu_sigmoid<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, gelu_sigmoid_func());
    }

    template<>
    void sigmoid<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, sigmoid_func());
    }

    template<>
    void swish<TARGET_ISA>(const float* x, float* y, dim_t size) {
      vectorized_unary_transform<TARGET_ISA>(x, y, size, swish_func());
    }

    template <CpuIsa ISA, typename T>
    void add(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::add(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void add(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::add);
    }

    template <CpuIsa ISA, typename T>
    void sub(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::sub);
    }

    template <CpuIsa ISA, typename T>
    void mul(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::mul(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void mul(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::mul);
    }

    template <CpuIsa ISA, typename T>
    void max(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::max(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void max(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::max);
    }

    template <CpuIsa ISA, typename T>
    void min(T a, const T* x, T* y, dim_t size) {
      auto vec_a = Vec<T, ISA>::load(a);
      vectorized_unary_transform<ISA>(x, y, size,
                                      [vec_a](vec_type<T, ISA> v) {
                                        return Vec<T, ISA>::min(v, vec_a);
                                      });
    }

    template <CpuIsa ISA, typename T>
    void min(const T* a, const T* b, T* c, dim_t size) {
      vectorized_binary_transform<ISA>(a, b, c, size, Vec<T, ISA>::min);
    }

    template <CpuIsa ISA, typename T>
    T reduce_sum(const T* x, dim_t size) {
      return vectorized_reduce_all<ISA>(x,
                                        size,
                                        static_cast<T>(0),
                                        Vec<T, ISA>::add,
                                        Vec<T, ISA>::reduce_add,
                                        Vec<T>::add);
    }

    template <CpuIsa ISA, typename T>
    T reduce_max(const T* x, dim_t size) {
      return vectorized_reduce_all<ISA>(x,
                                        size,
                                        std::numeric_limits<T>::lowest(),
                                        Vec<T, ISA>::max,
                                        Vec<T, ISA>::reduce_max,
                                        Vec<T>::max);
    }

    template <CpuIsa ISA, typename T>
    T reduce_amax(const T* x, dim_t size) {
      return vectorized_map_reduce_all<ISA>(x,
                                            size,
                                            static_cast<T>(0),
                                            Vec<T, ISA>::abs,
                                            Vec<T, ISA>::max,
                                            Vec<T, ISA>::reduce_max,
                                            Vec<T>::abs,
                                            Vec<T>::max);
    }

#define DECLARE_IMPL(T)                                                 \
    template void rcp<TARGET_ISA>(const T* x, T* y, dim_t size);        \
    template void add<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void add<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void sub<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void mul<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void mul<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void max<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void max<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template void min<TARGET_ISA>(T a, const T* x, T* y, dim_t size);   \
    template void min<TARGET_ISA>(const T* a, const T* b, T* c, dim_t size); \
    template T reduce_sum<TARGET_ISA>(const T* x, dim_t size);          \
    template T reduce_max<TARGET_ISA>(const T* x, dim_t size);          \
    template T reduce_amax<TARGET_ISA>(const T* x, dim_t size);

    DECLARE_ALL_TYPES(DECLARE_IMPL)


    template <>
    float reduce_logsumexp<TARGET_ISA>(const float* x, dim_t size) {
      using VecType = Vec<float, TARGET_ISA>;

      const auto x_max = reduce_max<TARGET_ISA>(x, size);
      const auto vec_x_max = VecType::load(x_max);

      const auto scalar_exp_func = [x_max](vec_type<float> v) {
        return Vec<float>::exp(Vec<float>::sub(v, x_max));
      };
      const auto vec_exp_func = [vec_x_max](vec_type<float, TARGET_ISA> v) {
        return VecType::exp(VecType::sub(v, vec_x_max));
      };

      const auto exp_sum = vectorized_map_reduce_all<TARGET_ISA>(
        x,
        size,
        static_cast<float>(0),
        vec_exp_func,
        VecType::add,
        VecType::reduce_add,
        scalar_exp_func,
        Vec<float>::add);

      return std::log(exp_sum) + x_max;
    }

    template<>
    void softmax<TARGET_ISA>(const float* input,
                             const int32_t* lengths,
                             float* output,
                             dim_t batch_size,
                             dim_t depth,
                             bool log) {
      using VecType = Vec<float, TARGET_ISA>;

      parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t offset = i * depth;
          const float* x = input + offset;
          float* y = output + offset;

          dim_t size = depth;
          if (lengths) {
            size = lengths[i];

            // Directly set 0 in output for out of range positions.
            for (dim_t j = size; j < depth; ++j) {
              y[j] = 0;
            }

            if (size == 0) {
              continue;
            }
          }

          const auto x_max = reduce_max<TARGET_ISA>(x, size);
          const auto vec_x_max = VecType::load(x_max);

          const auto scalar_exp_func = [x_max](vec_type<float> v) {
            return Vec<float>::exp(Vec<float>::sub(v, x_max));
          };
          const auto vec_exp_func = [vec_x_max](vec_type<float, TARGET_ISA> v) {
            return VecType::exp(VecType::sub(v, vec_x_max));
          };

          if (log) {
            const auto exp_sum = vectorized_map_reduce_all<TARGET_ISA>(
              x,
              size,
              static_cast<float>(0),
              vec_exp_func,
              VecType::add,
              VecType::reduce_add,
              scalar_exp_func,
              Vec<float>::add);
            add<TARGET_ISA>(-x_max - std::log(exp_sum), x, y, size);
          } else {
            vectorized_unary_transform<TARGET_ISA>(x, y, size, vec_exp_func);
            const auto exp_sum = reduce_sum<TARGET_ISA>(y, size);
            mul<TARGET_ISA>(static_cast<float>(1) / exp_sum, y, y, size);
          }
        }
      });
    }

    CT2_FFAST_MATH_BEGIN
    template<>
    void layer_norm<TARGET_ISA>(const float* input,
                                const float* gamma,
                                const float* beta,
                                float* output,
                                dim_t batch_size,
                                dim_t depth,
                                float epsilon) {
      parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const auto offset = i * depth;
          const auto* x = input + offset;
          auto* y = output + offset;

          // mean = sum(x)/n
          // rstd = 1/sqrt(var(x)) where var(x) = sum((x-mean)^2)/n = sum(x^2)/n - mean^2

          float sum = 0;
          float sum_squares = 0;

          for (dim_t j = 0; j < depth; ++j) {
            sum += x[j];
            sum_squares += x[j] * x[j];
          }

          const float mean = sum / depth;
          const float variance = std::max(sum_squares / depth - mean * mean, 0.f);
          const float rstd = 1.f / std::sqrt(variance + epsilon);

          for (dim_t j = 0; j < depth; ++j) {
            y[j] = (x[j] - mean) * rstd * gamma[j] + beta[j];
          }
        }
      });
    }

    template<>
    void layer_norm_axis<TARGET_ISA>(const float* input,
                                     const float* gamma,
                                     const float* beta,
                                     float* output,
                                     dim_t outer_size,
                                     dim_t axis_size,
                                     dim_t inner_size,
                                     float epsilon) {
      parallel_for(0, outer_size, 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          for (dim_t j = 0; j < inner_size; ++j) {
            // mean = sum(x)/n
            // rstd = 1/sqrt(var(x)) where var(x) = sum((x-mean)^2)/n = sum(x^2)/n - mean^2

            float sum = 0.f;
            float sum_squares = 0.f;

            for (dim_t k = 0; k < axis_size; ++k) {
              const dim_t index = i * axis_size * inner_size + k * inner_size + j;
              const float x = input[index];
              sum += x;
              sum_squares += x * x;
            }

            const float mean = sum / axis_size;
            const float variance = std::max(sum_squares / axis_size - mean * mean, 0.f);
            const float rstd = 1.f / std::sqrt(variance + epsilon);

            if (gamma && beta) {
              for (dim_t k = 0; k < axis_size; ++k) {
                const dim_t index = i * axis_size * inner_size + k * inner_size + j;
                output[index] = (input[index] - mean) * rstd * gamma[k] + beta[k];
              }
            } else {
              for (dim_t k = 0; k < axis_size; ++k) {
                const dim_t index = i * axis_size * inner_size + k * inner_size + j;
                output[index] = (input[index] - mean) * rstd;
              }
            }
          }
        }
      });
    }

    CT2_FFAST_MATH_END

  }
}
