#include "ctranslate2/allocator.h"

#ifdef _WIN32
#  include <malloc.h>
#else
#  include <cstdlib>
#endif


namespace ctranslate2 {
  namespace cpu {

    class AlignedAllocator : public Allocator {
    public:
      AlignedAllocator(size_t alignment)
        : _alignment(alignment)
      {
      }

      void* allocate(size_t size, int) override {
        void* ptr = nullptr;
#ifdef _WIN32
        ptr = _aligned_malloc(size, _alignment);
#else
        if (posix_memalign(&ptr, _alignment, size) != 0)
          ptr = nullptr;
#endif
        if (!ptr)
          throw std::runtime_error("aligned_alloc: failed to allocate memory");
        return ptr;
      }

      void free(void* ptr, int) override {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
      }

    private:
      size_t _alignment;
    };


  }

  template<>
  Allocator& get_allocator<Device::CPU>() {
    constexpr size_t alignment = 64;
    static cpu::AlignedAllocator allocator(alignment);
    return allocator;
  }

}
