cmake_minimum_required(VERSION 3.25)
project(halo_flash LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HALO_BUILD_TESTS "Build unit, property and stateful tests" ON)
option(HALO_FUZZ "Build libFuzzer targets (clang only)" OFF)
set(HALO_SANITIZE "" CACHE STRING "-fsanitize value, e.g. address,undefined or thread")

if(HALO_SANITIZE)
  add_compile_options(-fsanitize=${HALO_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${HALO_SANITIZE})
  if(HALO_SANITIZE MATCHES "undefined")
    add_compile_options(-fno-sanitize-recover=undefined)
  endif()
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)

if(HALO_FUZZ)
  # Instrument every library the fuzzers reach, not just the harnesses.
  add_compile_options(-fsanitize=fuzzer-no-link,address,undefined -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# --- core: file formats, quant storage, text -------------------------------
add_library(halo_core STATIC
  src/core/gguf.cpp
  src/core/gguf_writer.cpp
  src/core/quant.cpp
  src/core/json.cpp
  src/core/sampling.cpp
)
target_include_directories(halo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- text: Unicode, NFC, tokenizer, chat template, output parser -------------
add_library(halo_text STATIC
  src/text/unicode.cpp
  src/text/tokenizer.cpp
  src/text/chat_template.cpp
  src/text/tool_call_parser.cpp
)
target_link_libraries(halo_text PUBLIC halo_core)

# --- decide: System One decision API ---------------------------------------
add_library(halo_decide STATIC
  src/decide/systemone.cpp
  src/decide/decider.cpp
)
target_link_libraries(halo_decide PUBLIC halo_core halo_text halo_model)

# --- model: config, weight binding, n-gram table, CPU reference -------------
find_package(Threads REQUIRED)
add_library(halo_model STATIC
  src/model/config.cpp
  src/model/weights.cpp
  src/model/ngram.cpp
  src/model/cpu_ops.cpp
  src/model/reference.cpp
  src/model/lora.cpp
  src/model/qwen35.cpp
)
target_link_libraries(halo_model PUBLIC halo_core Threads::Threads)


# --- GPU: device layer (HIP, HRX), kernels, GPU models -----------------------
option(HALO_GPU "Build the GPU runtime layer and kernels" ON)
set(HALO_HRX_ROOT "" CACHE PATH "hrx-system checkout built into build/cmake (empty: no HRX backend)")
if(HALO_GPU)
  find_program(HIPCC hipcc REQUIRED)
  set(HALO_GEN ${CMAKE_BINARY_DIR}/gen)
  add_custom_command(
    OUTPUT ${HALO_GEN}/core.hsaco
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HALO_GEN}/kernels
    COMMAND ${HIPCC} -x hip --offload-arch=gfx1151 --offload-device-only --no-gpu-bundle-output
            -O3 -std=c++20 -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -o ${HALO_GEN}/core.hsaco ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/core.hip
    DEPENDS src/kernels/core.hip src/kernels/step_params.h
    COMMENT "Compiling GPU kernels (gfx1151 HSACO)")
  add_custom_command(
    OUTPUT ${HALO_GEN}/kernels/core_hsaco.cpp ${HALO_GEN}/kernels/core_hsaco.hpp
    COMMAND ${CMAKE_COMMAND} -DINPUT=${HALO_GEN}/core.hsaco -DNAME=core -DFUNC=CoreHsaco
            -DOUTPUT_CPP=${HALO_GEN}/kernels/core_hsaco.cpp -DOUTPUT_HPP=${HALO_GEN}/kernels/core_hsaco.hpp
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${HALO_GEN}/core.hsaco cmake/embed.cmake)
  add_library(halo_gpu STATIC
    src/rt/device.cpp
    src/rt/hip_device.cpp
    src/engine/qwen35_gpu.cpp
    src/engine/flash_next_gpu.cpp
    ${HALO_GEN}/kernels/core_hsaco.cpp)
  target_include_directories(halo_gpu PUBLIC ${HALO_GEN})
  target_compile_definitions(halo_gpu PUBLIC __HIP_PLATFORM_AMD__ HALO_WITH_HIP=1)
  target_link_libraries(halo_gpu PUBLIC halo_model amdhip64)
  target_link_libraries(halo_decide PUBLIC halo_gpu)
  if(HALO_HRX_ROOT)
    target_sources(halo_gpu PRIVATE src/rt/hrx_device.cpp)
    target_compile_definitions(halo_gpu PUBLIC HALO_WITH_HRX=1)
    target_include_directories(halo_gpu PRIVATE ${HALO_HRX_ROOT}/libhrx/include)
    set(HALO_HRX_LIBDIR ${HALO_HRX_ROOT}/build/cmake/libhrx/src/libhrx)
    target_link_libraries(halo_gpu PUBLIC ${HALO_HRX_LIBDIR}/libhrx.so)
    target_link_options(halo_gpu PUBLIC -Wl,-rpath,${HALO_HRX_LIBDIR})
  endif()
endif()

# --- tools ------------------------------------------------------------------
add_executable(gguf_inspect tools/gguf_inspect.cpp)
target_link_libraries(gguf_inspect PRIVATE halo_core)

# --- tests ------------------------------------------------------------------
if(HALO_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(hegel
    GIT_REPOSITORY https://github.com/hegeldev/hegel-cpp.git
    GIT_TAG v0.13.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest hegel)
  include(GoogleTest)

  function(halo_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE halo_core halo_text halo_model halo_decide GTest::gtest_main hegel)
    # Hegel's params structs are meant for partial designated initializers.
    target_compile_options(${name} PRIVATE -Wno-missing-designated-field-initializers)
    # Hegel's example database: failures found once are replayed first.
    gtest_discover_tests(${name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DISCOVERY_TIMEOUT 60)
  endfunction()

  halo_test(core_tests
    tests/core/gguf_test.cpp
    tests/core/quant_test.cpp
    tests/core/json_test.cpp
    tests/core/sampling_test.cpp)
  target_compile_definitions(core_tests PRIVATE
    HALO_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/core/data")
  halo_test(text_tests
    tests/text/tokenizer_test.cpp
    tests/text/chat_template_test.cpp
    tests/text/tool_call_parser_test.cpp)
  halo_test(decide_tests
    tests/decide/systemone_test.cpp
    tests/decide/decider_test.cpp)
  target_compile_definitions(decide_tests PRIVATE
    HALO_DECIDE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/decide/data")
  if(HALO_GPU)
    halo_test(gpu_tests tests/gpu/qwen35_gpu_test.cpp tests/gpu/quant_gpu_test.cpp tests/gpu/flash_next_gpu_test.cpp tests/gpu/select_gpu_test.cpp)
    target_link_libraries(gpu_tests PRIVATE halo_gpu)
    target_link_libraries(decide_tests PRIVATE halo_gpu)
    add_executable(halo_step_stats tools/step_stats.cpp)
    target_link_libraries(halo_step_stats PRIVATE halo_gpu)
  endif()
  halo_test(model_tests
    tests/model/reference_hf_test.cpp
    tests/model/ngram_table_test.cpp
    tests/model/lora_hf_test.cpp
    tests/model/qwen35_hf_test.cpp)
  target_compile_definitions(text_tests PRIVATE
    HALO_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/text/data")
endif()

# --- fuzzers ----------------------------------------------------------------
if(HALO_FUZZ)
  add_executable(gguf_fuzz fuzz/gguf_fuzz.cpp)
  target_compile_options(gguf_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(gguf_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(gguf_fuzz PRIVATE halo_core)
  # Writes valid GGUF images as a seed corpus: random bytes almost never get
  # past the magic and counts, so the fuzzer needs well-formed starting points.
  add_executable(tokenizer_fuzz fuzz/tokenizer_fuzz.cpp)
  target_link_options(tokenizer_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(tokenizer_fuzz PRIVATE halo_text)
  add_executable(tool_call_fuzz fuzz/tool_call_fuzz.cpp)
  target_link_options(tool_call_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(tool_call_fuzz PRIVATE halo_text)
  add_executable(json_fuzz fuzz/json_fuzz.cpp)
  target_link_options(json_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(json_fuzz PRIVATE halo_core)
  add_executable(gguf_seed fuzz/gguf_seed.cpp)
  target_link_libraries(gguf_seed PRIVATE halo_core)
endif()
