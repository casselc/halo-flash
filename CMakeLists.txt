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
)
target_include_directories(halo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

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
    target_link_libraries(${name} PRIVATE halo_core GTest::gtest_main hegel)
    # Hegel's params structs are meant for partial designated initializers.
    target_compile_options(${name} PRIVATE -Wno-missing-designated-field-initializers)
    # Hegel's example database: failures found once are replayed first.
    gtest_discover_tests(${name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      DISCOVERY_TIMEOUT 60)
  endfunction()

  halo_test(core_tests
    tests/core/gguf_test.cpp
    tests/core/quant_test.cpp)
endif()

# --- fuzzers ----------------------------------------------------------------
if(HALO_FUZZ)
  add_executable(gguf_fuzz fuzz/gguf_fuzz.cpp)
  target_compile_options(gguf_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(gguf_fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_libraries(gguf_fuzz PRIVATE halo_core)
  # Writes valid GGUF images as a seed corpus: random bytes almost never get
  # past the magic and counts, so the fuzzer needs well-formed starting points.
  add_executable(gguf_seed fuzz/gguf_seed.cpp)
  target_link_libraries(gguf_seed PRIVATE halo_core)
endif()
