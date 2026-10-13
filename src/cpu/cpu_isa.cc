#include "cpu_isa.h"

#include <stdexcept>

#include "env.h"

namespace ctranslate2 {
  namespace cpu {

    std::string isa_to_str(CpuIsa) {
      return "GENERIC";
    }

    static CpuIsa init_isa() {
      const std::string env_isa = read_string_from_env("CT2_FORCE_CPU_ISA");
      if (!env_isa.empty() && env_isa != "GENERIC")
        throw std::invalid_argument("Invalid CPU ISA: " + env_isa);
      return CpuIsa::GENERIC;
    }

    CpuIsa get_cpu_isa() {
      static const CpuIsa cpu_isa = init_isa();
      return cpu_isa;
    }

  }
}
