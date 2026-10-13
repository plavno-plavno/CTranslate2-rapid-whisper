#pragma once

#include "ctranslate2/types.h"

namespace ctranslate2 {
  namespace cpu {

    // This build has no CPU GEMM backend: models only run on CUDA devices and the CPU
    // only executes host-side bookkeeping (ids, beams, alignments).
    inline bool has_gemm_backend(ComputeType) {
      return false;
    }

    inline bool prefer_u8s8s32_gemm() {
      return false;
    }

    inline bool pack_gemm_weights(ComputeType) {
      return false;
    }

  }
}
