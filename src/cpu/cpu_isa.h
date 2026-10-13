#pragma once

#include <string>

namespace ctranslate2 {
  namespace cpu {

    enum class CpuIsa {
      GENERIC,
    };

    std::string isa_to_str(CpuIsa isa);

    // Returns the CPU ISA to dispatch to.
    CpuIsa get_cpu_isa();

  }
}

#define CPU_ISA_CASE(CPU_ISA, STMTS)            \
  case CPU_ISA: {                               \
    constexpr cpu::CpuIsa ISA = CPU_ISA;        \
    STMTS;                                      \
    break;                                      \
  }

#define CPU_ISA_DEFAULT(CPU_ISA, STMTS)                 \
  default: {                                            \
    constexpr cpu::CpuIsa ISA = CPU_ISA;                \
    STMTS;                                              \
    break;                                              \
  }

#define SINGLE_ARG(...) __VA_ARGS__
// Only the portable implementation is built: the CPU only runs host-side bookkeeping.
#define CPU_ISA_DISPATCH(STMTS)                               \
  switch (cpu::get_cpu_isa()) {                               \
    CPU_ISA_DEFAULT(cpu::CpuIsa::GENERIC, SINGLE_ARG(STMTS))  \
  }
