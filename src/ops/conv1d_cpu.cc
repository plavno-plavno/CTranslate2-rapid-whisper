#include "ctranslate2/ops/conv1d.h"

#include <stdexcept>

namespace ctranslate2 {
  namespace ops {

    template<>
    void
    Conv1D::compute<Device::CPU, float>(const StorageView&, const StorageView&, const StorageView*,
                                        StorageView&, const StorageView*) const {
      throw std::runtime_error("Conv1D is only implemented on CUDA in this build");
    }

  }
}
