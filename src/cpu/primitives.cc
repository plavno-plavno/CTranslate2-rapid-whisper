#include "ctranslate2/primitives.h"

#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>






#include "ctranslate2/allocator.h"
#include "cpu/backend.h"
#include "cpu/kernels.h"
#include "cpu/parallel.h"
#include "type_dispatch.h"

namespace ctranslate2 {

  static Allocator& allocator = get_allocator<Device::CPU>();

  template<>
  template <typename T>
  T primitives<Device::CPU>::at(const T* x, dim_t index) {
    return x[index];
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::fill(T* x, T a, dim_t size) {
    std::fill(x, x + size, a);
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size) {
    for (dim_t i = 0; i < size; i++, x += inc_x) {
      *x = a;
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::indexed_fill(T* x, T a, const int32_t* indices, dim_t num_indices) {
    for (dim_t i = 0; i < num_indices; ++i)
      x[indices[i]] = a;
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::copy(const T* x, T* y, dim_t size) {
    std::copy(x, x + size, y);
  }

  template<>
  template <typename U, typename V>
  void primitives<Device::CPU>::convert(const U* x, V* y, dim_t size) {
    std::copy(x, x + size, y);
  }

  template void primitives<Device::CPU>::convert(const float*, float16_t*, dim_t);
  template void primitives<Device::CPU>::convert(const float16_t*, float*, dim_t);
  template void primitives<Device::CPU>::convert(const float*, bfloat16_t*, dim_t);
  template void primitives<Device::CPU>::convert(const bfloat16_t*, float*, dim_t);
  template void primitives<Device::CPU>::convert(const float16_t*, bfloat16_t*, dim_t);
  template void primitives<Device::CPU>::convert(const bfloat16_t*, float16_t*, dim_t);

  template<>
  template <typename T>
  T primitives<Device::CPU>::sum(const T* array, dim_t size) {
    auto sum = T(0);
    CPU_ISA_DISPATCH((sum = cpu::reduce_sum<ISA>(array, size)));
    return sum;
  }

  template<>
  template <typename T>
  dim_t primitives<Device::CPU>::max_element(const T* array, dim_t size) {
    return std::distance(array, std::max_element(array, array + size));
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::max(const T* array, dim_t size) {
    auto max = T(0);
    CPU_ISA_DISPATCH((max = cpu::reduce_max<ISA>(array, size)));
    return max;
  }

  template<>
  template <typename T>
  T primitives<Device::CPU>::amax(const T* array, dim_t size) {
    auto max = T(0);
    CPU_ISA_DISPATCH((max = cpu::reduce_amax<ISA>(array, size)));
    return max;
  }

  template<>
  template<>
  float primitives<Device::CPU>::amax(const float* x, dim_t size) {
    float max = 0;
    CPU_ISA_DISPATCH((max = cpu::reduce_amax<ISA>(x, size)));
    return max;
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(T a, const T* x, T* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::add<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add(const T* a, const T* b, T* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::add<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::add(const float* a, const float* b, float* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::add<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = b_size / a_size;
    cpu::parallel_for(0, iter_size, 1, [&](dim_t begin, dim_t end) {
      for (dim_t i = begin; i < end; ++i) {
        const dim_t offset = i * a_size;
        add(a, b + offset, c + offset, a_size);
      }
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = a_size;
    const dim_t depth = b_size / a_size;
    cpu::parallel_for(0, iter_size, 1, [&](dim_t begin, dim_t end) {
      for (dim_t i = begin; i < end; ++i) {
        const dim_t offset = i * depth;
        add(a[i], b + offset, c + offset, depth);
      }
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::add_block_broadcast(const T* a, const T* b, T* c,
                                                    dim_t block, dim_t a_size, dim_t b_size) {
    const dim_t iter_size = b_size / block;
    cpu::parallel_for(0, iter_size, 1, [&](dim_t begin, dim_t end) {
      for (dim_t i = begin; i < end; ++i) {
        const T a_i = a[i % a_size];
        const dim_t offset = i * block;
        add(a_i, b + offset, c + offset, block);
      }
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::sub(const T* a, const T* b, T* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::sub<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::sub(const float* a, const float* b, float* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::sub<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::max(T a, const T* x, T* y, dim_t size){
    CPU_ISA_DISPATCH((cpu::max<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::max(const T* a, const T* b, T* c, dim_t size){
    CPU_ISA_DISPATCH((cpu::max<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::max(const float* a, const float* b, float* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::max<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::min(T a, const T* x, T* y, dim_t size){
    CPU_ISA_DISPATCH((cpu::min<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::min(const T* a, const T* b, T* c, dim_t size){
    CPU_ISA_DISPATCH((cpu::min<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::min(const float* a, const float* b, float* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::min<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(T a, const T* x, T* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, x, y, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::mul(float a, const float* x, float* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, x, y, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul(const T* a, const T* b, T* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, b, c, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::mul(const float* a, const float* b, float* c, dim_t size) {
    CPU_ISA_DISPATCH((cpu::mul<ISA>(a, b, c, size)));
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c,
                                                    dim_t a_size, dim_t b_size) {
    const dim_t iter_size = b_size / a_size;
    cpu::parallel_for(0, iter_size, 1, [&](dim_t begin, dim_t end) {
      for (dim_t i = begin; i < end; ++i) {
        const dim_t offset = i * a_size;
        mul(a, b + offset, c + offset, a_size);
      }
    });
  }

  template<>
  template<>
  void primitives<Device::CPU>::relu(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, cpu::GRAIN_SIZE,
                      [x, y](dim_t begin, dim_t end) {
                        max(float(0), x + begin, y + begin, end - begin);
                      });
  }

  template<>
  template<>
  void primitives<Device::CPU>::gelu(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, /*grain_size=*/512,
                      [x, y](dim_t begin, dim_t end) {
                        CPU_ISA_DISPATCH((cpu::gelu<ISA>(x + begin, y + begin, end - begin)));
                      });
  }

  template<>
  template<>
  void primitives<Device::CPU>::gelu_tanh(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, /*grain_size=*/512,
                      [x, y](dim_t begin, dim_t end) {
                        CPU_ISA_DISPATCH((cpu::gelu_tanh<ISA>(x + begin, y + begin, end - begin)));
                      });
  }

  template<>
  template<>
  void primitives<Device::CPU>::gelu_sigmoid(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, /*grain_size=*/512,
                      [x, y](dim_t begin, dim_t end) {
                        CPU_ISA_DISPATCH((cpu::gelu_sigmoid<ISA>(x + begin, y + begin, end - begin)));
                      });
  }

  template<>
  template<>
  void primitives<Device::CPU>::sigmoid(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, cpu::GRAIN_SIZE / 10,
                      [x, y](dim_t begin, dim_t end) {
                        CPU_ISA_DISPATCH((cpu::sigmoid<ISA>(x + begin, y + begin, end - begin)));
                      });
  }

  template<>
  template<>
  void primitives<Device::CPU>::swish(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, cpu::GRAIN_SIZE / 10,
                      [x, y](dim_t begin, dim_t end) {
                        CPU_ISA_DISPATCH((cpu::swish<ISA>(x + begin, y + begin, end - begin)));
                      });
  }

  template<>
  template<>
  float primitives<Device::CPU>::logsumexp(const float* x, dim_t size) {
    float result = 0;
    CPU_ISA_DISPATCH((result = cpu::reduce_logsumexp<ISA>(x, size)));
    return result;
  }

  template<>
  template<>
  void primitives<Device::CPU>::exp(const float* x, float* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::exp<ISA>(x, y, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::log(const float* x, float* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::log<ISA>(x, y, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::cos(const float* x, float* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::cos<ISA>(x, y, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::sin(const float* x, float* y, dim_t size) {
    CPU_ISA_DISPATCH((cpu::sin<ISA>(x, y, size)));
  }

  template<>
  template<>
  void primitives<Device::CPU>::tanh(const float* x, float* y, dim_t size) {
    cpu::parallel_for(0, size, /*grain_size=*/512,
                      [x, y](dim_t begin, dim_t end) {
                        CPU_ISA_DISPATCH((cpu::tanh<ISA>(x + begin, y + begin, end - begin)));
                      });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::penalize_previous_tokens(T* scores,
                                                         const T* previous_scores,
                                                         const int32_t* previous_ids,
                                                         T penalty,
                                                         dim_t batch_size,
                                                         dim_t length,
                                                         dim_t vocabulary_size) {
    cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
      for (dim_t i = begin; i < end; ++i) {
        for (dim_t j = 0; j < length; ++j) {
          const dim_t read_index = i * length + j;
          const dim_t write_index = i * vocabulary_size + previous_ids[read_index];
          const auto score = previous_scores[read_index];
          scores[write_index] = (score < T(0) ? score * penalty : score / penalty);
        }
      }
    });
  }

  template<>
  void primitives<Device::CPU>::prepare_length_mask(const int32_t* lengths,
                                                    dim_t batch_size,
                                                    dim_t num_heads,
                                                    dim_t num_queries,
                                                    bool mask_future,
                                                    bool multi_query,
                                                    int32_t* mask) {
    for (dim_t b = 0; b < batch_size; ++b) {
      const auto length = lengths[b];
      auto* batch_mask = mask + b * num_heads * num_queries;
      for (dim_t i = 0; i < num_heads * num_queries; ++i) {
        batch_mask[i] = (mask_future
                         ? std::min(length,
                                    int32_t((multi_query ? i / num_heads : i % num_queries) + 1))
                         : length);
      }
    }
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_2d(const T* a, const dim_t* dims, T* b) {
    cpu::parallel_for(0, dims[0], 1, [&](dim_t begin, dim_t end) {
      for (dim_t i0 = begin; i0 < end; ++i0) {
        for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
          b[i1 * dims[0] + i0] = a[i0 * dims[1] + i1];
        }
      }
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_3d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    dim_t perm_ind[3];
    for (dim_t i = 0; i < 3; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[3] = {dims[1] * dims[2], dims[2], 1};
    const dim_t b_stride[3] = {dims[perm[1]] * dims[perm[2]], dims[perm[2]], 1};
    const dim_t perm_b_stride[3] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]]};

    cpu::parallel_for(0, dims[0], 1, [&](dim_t begin, dim_t end) {
      for (dim_t i0 = begin; i0 < end; ++i0) {
        for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
          for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
            const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                               i2 * perm_b_stride[2]);
            const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                               i2 * a_stride[2]);
            b[b_i] = a[a_i];
          }
        }
      }
    });
  }

  template<>
  template <typename T>
  void primitives<Device::CPU>::transpose_4d(const T* a,
                                             const dim_t* dims,
                                             const dim_t* perm,
                                             T* b) {
    if (perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3) {
      // Optimize the permutation used in multi-head attention.
      const dim_t r1 = dims[2];
      const dim_t r2 = dims[1];
      const dim_t depth = dims[3];

      cpu::parallel_for(0, dims[0], 1, [&](dim_t begin, dim_t end) {
        for (dim_t i = begin; i < end; ++i) {
          const dim_t offset = i * r1 * r2;
          for (dim_t j = 0; j < r1 * r2; ++j) {
            const dim_t a_offset = depth * (offset + j);
            const dim_t b_offset = depth * (offset + j / r1 + (j % r1) * r2);
            copy(a + a_offset, b + b_offset, depth);
          }
        }
      });

      return;
    }

    dim_t perm_ind[4];
    for (dim_t i = 0; i < 4; ++i)
      perm_ind[perm[i]] = i;
    const dim_t a_stride[4] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    const dim_t b_stride[4] = {dims[perm[1]] * dims[perm[2]] * dims[perm[3]],
                               dims[perm[2]] * dims[perm[3]], dims[perm[3]], 1};
    const dim_t perm_b_stride[4] = {b_stride[perm_ind[0]], b_stride[perm_ind[1]],
                                    b_stride[perm_ind[2]], b_stride[perm_ind[3]]};

    cpu::parallel_for(0, dims[0], 1, [&](dim_t begin, dim_t end) {
      for (dim_t i0 = begin; i0 < end; ++i0) {
        for (dim_t i1 = 0; i1 < dims[1]; ++i1) {
          for (dim_t i2 = 0; i2 < dims[2]; ++i2) {
            for (dim_t i3 = 0; i3 < dims[3]; ++i3) {
              const dim_t b_i = (i0 * perm_b_stride[0] + i1 * perm_b_stride[1] +
                                 i2 * perm_b_stride[2] + i3 * perm_b_stride[3]);
              const dim_t a_i = (i0 * a_stride[0] + i1 * a_stride[1] +
                                 i2 * a_stride[2] + i3 * a_stride[3]);
              b[b_i] = a[a_i];
            }
          }
        }
      }
    });
  }



  template<>
  template<>
  dim_t primitives<Device::CPU>::gemm_pack_b(const float* b,
                                             const bool transpose_b,
                                             const dim_t k,
                                             const dim_t n,
                                             const float alpha,
                                             float* dest) {
    (void)b;
    (void)transpose_b;
    (void)k;
    (void)n;
    (void)alpha;
    (void)dest;
    return 0;
  }

  template<>
  template<>
  dim_t primitives<Device::CPU>::gemm_pack_b(const int16_t* b,
                                             const bool transpose_b,
                                             const dim_t k,
                                             const dim_t n,
                                             const float,
                                             int16_t* dest) {
    (void)b;
    (void)transpose_b;
    (void)k;
    (void)n;
    (void)dest;
    return 0;
  }

  template<>
  template<>
  dim_t primitives<Device::CPU>::gemm_pack_b(const int8_t* b,
                                             const bool transpose_b,
                                             const dim_t k,
                                             const dim_t n,
                                             const float,
                                             int8_t* dest) {
    (void)b;
    (void)transpose_b;
    (void)k;
    (void)n;
    (void)dest;
    return 0;
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(bool a_is_packed, bool b_is_packed,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const float* a, dim_t lda,
                                     const float* b, dim_t ldb,
                                     float beta,
                                     float* c, dim_t ldc,
                                     const float*) {
    (void)a_is_packed;
    (void)b_is_packed;

    throw std::runtime_error("No SGEMM backend on CPU");
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm(bool a_is_packed, bool b_is_packed,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const int16_t* a, dim_t lda,
                                     const int16_t* b, dim_t ldb,
                                     float beta,
                                     int32_t* c, dim_t ldc,
                                     const int32_t*) {
    (void)a_is_packed;
    (void)b_is_packed;
    (void)transpose_a;
    (void)transpose_b;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)a;
    (void)lda;
    (void)b;
    (void)ldb;
    (void)beta;
    (void)c;
    (void)ldc;

    throw std::runtime_error("No INT16 GEMM backend on CPU");
  }


  template<>
  void primitives<Device::CPU>::compute_u8_compensation(const int8_t* b,
                                                        bool transpose_b,
                                                        dim_t k,
                                                        dim_t n,
                                                        float alpha,
                                                        int32_t* compensation) {
    cpu::parallel_for(0, n, 1, [&](dim_t begin, dim_t end) {
      for (dim_t i = begin; i < end; ++i) {
        int32_t val = 0;

        if (transpose_b) {
          const int8_t* row = b + i * k;
          val = std::accumulate(row, row + k, static_cast<int32_t>(0));
        } else {
          for (dim_t j = 0; j < k; ++j) {
            val += b[j * n + i];
          }
        }

        if (alpha != 1) {
          val = std::nearbyintf(static_cast<float>(val) * alpha * -128.f);
        } else {
          val *= -128;
        }

        compensation[i] = val;
      }
    });
  }


  template<>
  template<>
  void primitives<Device::CPU>::gemm(bool a_is_packed, bool b_is_packed,
                                     bool transpose_a, bool transpose_b,
                                     dim_t m, dim_t n, dim_t k,
                                     float alpha,
                                     const int8_t* a, dim_t lda,
                                     const int8_t* b, dim_t ldb,
                                     float beta,
                                     int32_t* c, dim_t ldc,
                                     const int32_t* a_shift_compensation) {
    (void)a_is_packed;
    (void)b_is_packed;

    throw std::runtime_error("No INT8 GEMM backend for CPU");
  }

  template<>
  template<>
  void primitives<Device::CPU>::gemm_batch_strided(bool transpose_a, bool transpose_b,
                                                   dim_t m, dim_t n, dim_t k,
                                                   float alpha,
                                                   const float* a, dim_t lda, dim_t stridea,
                                                   const float* b, dim_t ldb, dim_t strideb,
                                                   float beta,
                                                   float* c, dim_t ldc, dim_t stridec,
                                                   dim_t batch_size) {
    cpu::parallel_for(0, batch_size, 1, [&](dim_t begin, dim_t end) {
      for (dim_t i = begin; i < end; ++i) {
        const float* a_i = a + (i * stridea);
        const float* b_i = b + (i * strideb);
        float* c_i = c + (i * stridec);

        gemm(/*a_is_packed=*/false, /*b_is_packed=*/false,
             transpose_a, transpose_b,
             m, n, k,
             alpha,
             a_i, lda,
             b_i, ldb,
             beta,
             c_i, ldc);
      }
    });
  }


#define DECLARE_IMPL(T)                                                 \
  template T                                                            \
  primitives<Device::CPU>::at(const T* x, dim_t index);                 \
  template void                                                         \
  primitives<Device::CPU>::fill(T* x, T a, dim_t size);                 \
  template void                                                         \
  primitives<Device::CPU>::strided_fill(T* x, T a, dim_t inc_x, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::indexed_fill(T*, T, const int32_t*, dim_t);  \
  template void                                                         \
  primitives<Device::CPU>::copy(const T* x, T* y, dim_t size);          \
  template T                                                            \
  primitives<Device::CPU>::sum(const T* array, dim_t size);             \
  template dim_t                                                        \
  primitives<Device::CPU>::max_element(const T* array, dim_t size);     \
  template T                                                            \
  primitives<Device::CPU>::max(const T* array, dim_t size);             \
  template void                                                         \
  primitives<Device::CPU>::add(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CPU>::add_batch_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::add_depth_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::add_block_broadcast(const T* a, const T* b, T* c, \
                                               dim_t block, dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::min(T a, const T* x, T* y, dim_t size);      \
  template void                                                         \
  primitives<Device::CPU>::max(T a, const T* x, T* y, dim_t size);     \
  template void                                                         \
  primitives<Device::CPU>::mul_batch_broadcast(const T* a, const T* b, T* c, \
                                               dim_t a_size, dim_t b_size); \
  template void                                                         \
  primitives<Device::CPU>::penalize_previous_tokens(T*,                 \
                                                    const T*,           \
                                                    const int32_t*,     \
                                                    T,                  \
                                                    dim_t,              \
                                                    dim_t,              \
                                                    dim_t);             \
  template void                                                         \
  primitives<Device::CPU>::transpose_2d(const T* a,                     \
                                        const dim_t* dims,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::transpose_3d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);                          \
  template void                                                         \
  primitives<Device::CPU>::transpose_4d(const T* a,                     \
                                        const dim_t* dims,              \
                                        const dim_t* perm,              \
                                        T* b);

  DECLARE_ALL_TYPES(DECLARE_IMPL)

#define DECLARE_IMPL_NO_FLOAT(T)                                        \
  template T                                                            \
  primitives<Device::CPU>::amax(const T* array, dim_t size);            \
  template void                                                         \
  primitives<Device::CPU>::add(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::sub(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::min(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::max(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul(const T* a, const T* b, T* c, dim_t size); \
  template void                                                         \
  primitives<Device::CPU>::mul(T a, const T* x, T* y, dim_t size);

  DECLARE_IMPL_NO_FLOAT(int8_t)
  DECLARE_IMPL_NO_FLOAT(int16_t)
  DECLARE_IMPL_NO_FLOAT(int32_t)
  DECLARE_IMPL_NO_FLOAT(float16_t)
  DECLARE_IMPL_NO_FLOAT(bfloat16_t)

}
