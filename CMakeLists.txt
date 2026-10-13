cmake_minimum_required(VERSION 3.7)

# Set policy for setting the MSVC runtime library for static MSVC builds
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

project(ctranslate2)

option(WITH_CUDA "Compile with CUDA backend" OFF)
option(WITH_CUDNN "Compile with cuDNN backend" OFF)
option(CUDA_DYNAMIC_LOADING "Dynamically load CUDA libraries at runtime" OFF)
option(ENABLE_PROFILING "Compile with profiling support" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(ENABLE_ADDRESS_SANITIZER "ASAN" OFF)
option(ENABLE_THREAD_SANITIZER "TSAN" OFF)

MESSAGE(STATUS "Compiler Id: ${CMAKE_CXX_COMPILER_ID}")
MESSAGE(STATUS "Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

if(ENABLE_PROFILING)
  message(STATUS "Enable profiling support")
  add_definitions(-DCT2_ENABLE_PROFILING)
endif()


# Set Release build type by default to get sane performance.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

# Set CXX flags.
set(CMAKE_CXX_STANDARD 17)

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 10.13)
endif()


# Read version from version.py
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/python/ctranslate2/version.py VERSION_FILE)
foreach(line IN LISTS VERSION_FILE)
  if (line MATCHES "__version__")
    string(REGEX MATCH "[0-9.]+" CTRANSLATE2_VERSION ${line})
    break()
  endif()
endforeach()

if(NOT CTRANSLATE2_VERSION)
  message(FATAL_ERROR "Version can't be read from version.py")
endif()

string(REPLACE "." ";" CTRANSLATE2_VERSION_LIST ${CTRANSLATE2_VERSION})
list(GET CTRANSLATE2_VERSION_LIST 0 CTRANSLATE2_MAJOR_VERSION)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /d2FH4-")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

if(WIN32)
  add_compile_definitions(_USE_MATH_DEFINES) # required for M_PI
  if(BUILD_SHARED_LIBS)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  else()
    if(CMAKE_VERSION VERSION_LESS "3.15.0")
      message(FATAL_ERROR "Use CMake 3.15 or later when setting BUILD_SHARED_LIBS to OFF")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
elseif(NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,noexecstack")
endif()

find_package(Threads)
add_subdirectory(third_party/spdlog EXCLUDE_FROM_ALL)

set(PRIVATE_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  )
set(SOURCES
  src/allocator.cc
  src/batch_reader.cc
  src/cpu/allocator.cc
  src/cpu/cpu_isa.cc
  src/cpu/kernels.cc
  src/cpu/parallel.cc
  src/cpu/primitives.cc
  src/decoding.cc
  src/decoding_utils.cc
  src/devices.cc
  src/dtw.cc
  src/env.cc
  src/filesystem.cc
  src/layers/attention_layer.cc
  src/layers/attention.cc
  src/layers/common.cc
  src/layers/decoder.cc
  src/layers/transformer.cc
  src/layers/whisper.cc
  src/logging.cc
  src/models/model.cc
  src/models/model_factory.cc
  src/models/model_reader.cc
  src/models/whisper.cc
  src/ops/activation.cc
  src/ops/add.cc
  src/ops/alibi_add.cc
  src/ops/alibi_add_cpu.cc
  src/ops/bias_add.cc
  src/ops/bias_add_cpu.cc
  src/ops/concat.cc
  src/ops/concat_split_slide_cpu.cc
  src/ops/conv1d.cc
  src/ops/conv1d_cpu.cc
  src/ops/cos.cc
  src/ops/gather.cc
  src/ops/gather_cpu.cc
  src/ops/gelu.cc
  src/ops/gemm.cc
  src/ops/gumbel_max.cc
  src/ops/gumbel_max_cpu.cc
  src/ops/layer_norm.cc
  src/ops/layer_norm_cpu.cc
  src/ops/log.cc
  src/ops/matmul.cc
  src/ops/mean.cc
  src/ops/mean_cpu.cc
  src/ops/median_filter.cc
  src/ops/median_filter_cpu.cc
  src/ops/min_max.cc
  src/ops/mul.cc
  src/ops/multinomial.cc
  src/ops/multinomial_cpu.cc
  src/ops/relu.cc
  src/ops/rms_norm.cc
  src/ops/rms_norm_cpu.cc
  src/ops/rotary.cc
  src/ops/rotary_cpu.cc
  src/ops/sin.cc
  src/ops/softmax.cc
  src/ops/softmax_cpu.cc
  src/ops/split.cc
  src/ops/slide.cc
  src/ops/sub.cc
  src/ops/sigmoid.cc
  src/ops/swish.cc
  src/ops/tanh.cc
  src/ops/tile.cc
  src/ops/tile_cpu.cc
  src/ops/topk.cc
  src/ops/topk_cpu.cc
  src/ops/topp_mask.cc
  src/ops/topp_mask_cpu.cc
  src/ops/transpose.cc
  src/ops/sum.cc
  src/padder.cc
  src/profiler.cc
  src/random.cc
  src/sampling.cc
  src/storage_view.cc
  src/thread_pool.cc
  src/types.cc
  src/utils.cc
  src/vocabulary.cc
)
set(CUDA_SOURCES
  src/cuda/allocator.cc
  src/cuda/primitives.cu
  src/cuda/random.cu
  src/cuda/utils.cc
  src/ops/alibi_add_gpu.cu
  src/ops/bias_add_gpu.cu
  src/ops/concat_split_slide_gpu.cu
  src/ops/conv1d_gpu.cu
  src/ops/gather_gpu.cu
  src/ops/gumbel_max_gpu.cu
  src/ops/layer_norm_gpu.cu
  src/ops/mean_gpu.cu
  src/ops/median_filter_gpu.cu
  src/ops/multinomial_gpu.cu
  src/ops/rms_norm_gpu.cu
  src/ops/rotary_gpu.cu
  src/ops/softmax_gpu.cu
  src/ops/tile_gpu.cu
  src/ops/topk_gpu.cu
  src/ops/topp_mask_gpu.cu
)
set(LIBRARIES
  ${CMAKE_THREAD_LIBS_INIT}
  spdlog::spdlog_header_only
  )

# Position independent code for the shared library and its static dependencies.
if(BUILD_SHARED_LIBS)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# sanitizers
IF (ENABLE_ADDRESS_SANITIZER AND (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG"))
  MESSAGE (STATUS "ENABLE_ADDRESS_SANITIZER: TRUE")
  set(ASAN_FLAGS " -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-common")
  string(APPEND CMAKE_C_FLAGS ${ASAN_FLAGS})
  string(APPEND CMAKE_CXX_FLAGS ${ASAN_FLAGS})
  add_link_options(-fsanitize=address)
ELSEIF (ENABLE_THREAD_SANITIZER AND (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "DEBUG"))
  MESSAGE (STATUS "ENABLE_THREAD_SANITIZER: TRUE")
  set(TSAN_FLAGS " -fsanitize=thread")
  string(APPEND CMAKE_C_FLAGS ${TSAN_FLAGS})
  string(APPEND CMAKE_CXX_FLAGS ${TSAN_FLAGS})
  add_link_options(-fsanitize=thread)
ELSEIF (ENABLE_ADDRESS_SANITIZER OR ENABLE_THREAD_SANITIZER)
  MESSAGE(FATAL_ERROR "SANITIZER requires Debug build type")
ENDIF ()

if (WITH_CUDA)
  find_package(CUDA 11.0 REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)

  add_definitions(-DCT2_WITH_CUDA)
  if(MSVC)
    if(BUILD_SHARED_LIBS)
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/MD$<$<CONFIG:Debug>:d>")
    else()
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/MT$<$<CONFIG:Debug>:d>")
    endif()
  endif()
  list(APPEND CUDA_NVCC_FLAGS "-std=c++17")

  if(NOT CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST "Auto")
  elseif(CUDA_ARCH_LIST STREQUAL "Common")
    set(CUDA_ARCH_LIST ${CUDA_COMMON_GPU_ARCHITECTURES})
    # Keep deprecated but not yet dropped Compute Capabilities.
    if(CUDA_VERSION_MAJOR EQUAL 11)
      list(INSERT CUDA_ARCH_LIST 0 "3.5" "5.0")
    endif()
    list(REMOVE_DUPLICATES CUDA_ARCH_LIST)
  endif()

  if(CUDA_ARCH_FLAGS_OVERRIDE)
    # FindCUDA in older CMake cannot parse two-digit major archs (e.g. 12.0 for Blackwell).
    set(ARCH_FLAGS ${CUDA_ARCH_FLAGS_OVERRIDE})
  else()
    cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})
  endif()
  list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  message(STATUS "NVCC host compiler: ${CUDA_HOST_COMPILER}")
  message(STATUS "NVCC compilation flags: ${CUDA_NVCC_FLAGS}")

  # We should ensure that the Thrust include directories appear before
  # -I/usr/local/cuda/include for both GCC and NVCC, so that the headers
  # are coming from the submodule and not the system.
  set(THRUST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/thrust/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/thrust/thrust
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/thrust/libcudacxx/include
    )
  cuda_include_directories(${THRUST_INCLUDE_DIRS})
  list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${THRUST_INCLUDE_DIRS})


  if(WITH_CUDNN)
    # Find cuDNN includes.
    find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS ${CUDA_TOOLKIT_ROOT_DIR}/include)
    if(CUDNN_INCLUDE_DIR)
      message(STATUS "Found cuDNN include directory: ${CUDNN_INCLUDE_DIR}")
    else()
      message(FATAL_ERROR "cuDNN include directory not found")
    endif()

    # Find cuDNN libraries.
    find_library(CUDNN_LIBRARIES
      NAMES cudnn
      HINTS
      ${CUDA_TOOLKIT_ROOT_DIR}/lib
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
    )
    if(CUDNN_LIBRARIES)
      message(STATUS "Found cuDNN libraries: ${CUDNN_LIBRARIES}")
    else()
      message(FATAL_ERROR "cuDNN libraries not found")
    endif()

    # libcudnn.so is a shim layer that dynamically loads the correct library at runtime,
    # so we explicitly link against it even with CUDA_DYNAMIC_LOADING.
    list(APPEND PRIVATE_INCLUDE_DIRECTORIES ${CUDNN_INCLUDE_DIR})
    list(APPEND LIBRARIES ${CUDNN_LIBRARIES})
    add_definitions(-DCT2_WITH_CUDNN)
    list(APPEND CUDA_SOURCES src/ops/conv1d_cudnn_gpu.cu)
    list(REMOVE_ITEM CUDA_SOURCES src/ops/conv1d_gpu.cu)
  endif()

  if(CUDA_DYNAMIC_LOADING)
    list(APPEND SOURCES src/cuda/cublas_stub.cc)
  else()
    list(APPEND LIBRARIES ${CUDA_CUBLAS_LIBRARIES})
  endif()
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  cuda_add_library(${PROJECT_NAME}
    ${SOURCES}
    ${CUDA_SOURCES}
  )


elseif(WITH_CUDNN)
  message(FATAL_ERROR "WITH_CUDNN=ON requires WITH_CUDA=ON")
else()
  add_library(${PROJECT_NAME} ${SOURCES})
endif()

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME})
set_property(TARGET ${PROJECT_NAME} PROPERTY VERSION ${CTRANSLATE2_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY SOVERSION ${CTRANSLATE2_MAJOR_VERSION})
set_property(TARGET ${PROJECT_NAME} PROPERTY
  INTERFACE_${PROJECT_NAME}_MAJOR_VERSION ${CTRANSLATE2_MAJOR_VERSION})
set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING ${PROJECT_NAME}_MAJOR_VERSION
)

list(APPEND LIBRARIES ${CMAKE_DL_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})
target_include_directories(${PROJECT_NAME} BEFORE
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${PRIVATE_INCLUDE_DIRECTORIES}
)

include(GNUInstallDirs)

install(
  TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h*"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${CTRANSLATE2_VERSION}
  COMPATIBILITY AnyNewerVersion
)

if(BUILD_SHARED_LIBS)
  export(EXPORT ${PROJECT_NAME}Targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}Targets.cmake"
    NAMESPACE CTranslate2::
  )
endif()

configure_file(cmake/${PROJECT_NAME}Config.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}Config.cmake"
  COPYONLY
)

set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

if(BUILD_SHARED_LIBS)
  install(EXPORT ${PROJECT_NAME}Targets
    FILE
      ${PROJECT_NAME}Targets.cmake
    NAMESPACE
      CTranslate2::
    DESTINATION
      ${ConfigPackageLocation}
  )
endif()

install(
  FILES
    cmake/${PROJECT_NAME}Config.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)
