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
  src/core/lowbit.cpp
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

# --- server: OpenAI request and response formats -----------------------------
add_library(halo_server STATIC
  src/server/openai_request.cpp
  src/server/openai_response.cpp
  src/server/http.cpp
  src/server/openai_service.cpp
  src/server/systemone_service.cpp
  src/server/metrics.cpp
)
target_link_libraries(halo_server PUBLIC Threads::Threads)
target_link_libraries(halo_server PUBLIC halo_core halo_text halo_decide)

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
  # One ROCm install for compiler, headers and runtime: the HIP runtime and
  # headers are taken from the install hipcc belongs to, never from wherever
  # the linker or the host compiler would find them first (mixing two ROCm
  # versions builds and runs, and silently detaches profilers built for one).
  get_filename_component(_halo_hipcc_real "${HIPCC}" REALPATH)
  get_filename_component(HALO_ROCM_PREFIX "${_halo_hipcc_real}" DIRECTORY)
  get_filename_component(HALO_ROCM_PREFIX "${HALO_ROCM_PREFIX}" DIRECTORY)
  find_library(HALO_AMDHIP amdhip64 PATHS "${HALO_ROCM_PREFIX}/lib" "${HALO_ROCM_PREFIX}/lib/x86_64-linux-gnu"
               NO_DEFAULT_PATH REQUIRED)
  find_path(HALO_HIP_INCLUDE hip/hip_runtime.h PATHS "${HALO_ROCM_PREFIX}/include" NO_DEFAULT_PATH REQUIRED)
  execute_process(COMMAND ${HIPCC} --version OUTPUT_VARIABLE _halo_hip_version ERROR_QUIET)
  string(REGEX MATCH "HIP version: [^\n]*" HALO_HIP_VERSION "${_halo_hip_version}")
  set(HALO_TOOLCHAIN_ID "$ENV{HALO_TOOLCHAIN}")
  if(NOT HALO_TOOLCHAIN_ID)
    set(HALO_TOOLCHAIN_ID "system")
  endif()
  message(STATUS "halo-flash ROCm: ${HALO_ROCM_PREFIX} (${HALO_HIP_VERSION}), runtime ${HALO_AMDHIP}, toolchain ${HALO_TOOLCHAIN_ID}")
  file(WRITE ${HALO_GEN}/toolchain.h.in
       "#pragma once\n#define HALO_TOOLCHAIN \"${HALO_TOOLCHAIN_ID}; ${HALO_HIP_VERSION}; hipcc ${_halo_hipcc_real}; runtime ${HALO_AMDHIP}\"\n")
  configure_file(${HALO_GEN}/toolchain.h.in ${HALO_GEN}/halo_toolchain.h COPYONLY)
  # The kernels may be compiled by another device compiler than the runtime's
  # hipcc (e.g. a TheRock nightly clang with its device libraries), the rest
  # of the build unchanged:
  #   -DHALO_HSACO_COMPILER=<clang> "-DHALO_HSACO_FLAGS=--hip-path=<hip>;--rocm-device-lib-path=<bitcode>"
  set(HALO_HSACO_COMPILER "${HIPCC}" CACHE FILEPATH "Compiler for the GPU kernels (HSACO)")
  set(HALO_HSACO_FLAGS "" CACHE STRING "Extra flags for HALO_HSACO_COMPILER (a CMake list)")
  # Hardware probes (standalone HIP program, docs/HARDWARE.md).
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/halo_hwprobe
    COMMAND ${HIPCC} -O3 -std=c++20 --offload-arch=gfx1151 ${CMAKE_CURRENT_SOURCE_DIR}/tools/hwprobe.hip
            -o ${CMAKE_BINARY_DIR}/halo_hwprobe
    DEPENDS tools/hwprobe.hip
    COMMENT "Building halo_hwprobe")
  add_custom_target(hwprobe ALL DEPENDS ${CMAKE_BINARY_DIR}/halo_hwprobe)
  add_custom_command(
    OUTPUT ${HALO_GEN}/core.hsaco
    COMMAND ${CMAKE_COMMAND} -E make_directory ${HALO_GEN}/kernels
    COMMAND ${HALO_HSACO_COMPILER} -x hip --offload-arch=gfx1151 --offload-device-only --no-gpu-bundle-output
            ${HALO_HSACO_FLAGS} -O3 -std=c++20 -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -o ${HALO_GEN}/core.hsaco ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/core.hip
    DEPENDS src/kernels/core.hip src/kernels/step_params.h src/kernels/span_trace.h
    COMMENT "Compiling GPU kernels (gfx1151 HSACO)")
  # The same kernels with span tracing compiled in (src/kernels/span_trace.h): loaded from the file by the
  # tools that trace (halo_gemm_bench trace), never embedded in the engine.
  add_custom_command(
    OUTPUT ${HALO_GEN}/core_trace.hsaco
    COMMAND ${HALO_HSACO_COMPILER} -x hip --offload-arch=gfx1151 --offload-device-only --no-gpu-bundle-output
            ${HALO_HSACO_FLAGS} -O3 -std=c++20 -DHALO_TRACE -I${CMAKE_CURRENT_SOURCE_DIR}/src
            -o ${HALO_GEN}/core_trace.hsaco ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/core.hip
    DEPENDS src/kernels/core.hip src/kernels/step_params.h src/kernels/span_trace.h
    COMMENT "Building core_trace.hsaco (span tracing)")
  add_custom_target(core_trace ALL DEPENDS ${HALO_GEN}/core_trace.hsaco)
  add_custom_command(
    OUTPUT ${HALO_GEN}/kernels/core_hsaco.cpp ${HALO_GEN}/kernels/core_hsaco.hpp
    COMMAND ${CMAKE_COMMAND} -DINPUT=${HALO_GEN}/core.hsaco -DNAME=core -DFUNC=CoreHsaco
            -DOUTPUT_CPP=${HALO_GEN}/kernels/core_hsaco.cpp -DOUTPUT_HPP=${HALO_GEN}/kernels/core_hsaco.hpp
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${HALO_GEN}/core.hsaco cmake/embed.cmake)
  add_library(halo_gpu STATIC
    src/rt/device.cpp
    src/rt/hip_device.cpp
    src/rt/file_upload.cpp
    src/engine/qwen35_gpu.cpp
    src/engine/flash_next_gpu.cpp
    src/engine/prefill_chunk.cpp
    ${HALO_GEN}/kernels/core_hsaco.cpp)
  target_include_directories(halo_gpu PUBLIC ${HALO_GEN})
  target_compile_definitions(halo_gpu PUBLIC __HIP_PLATFORM_AMD__ HALO_WITH_HIP=1)
  target_include_directories(halo_gpu PUBLIC ${HALO_HIP_INCLUDE})
  target_link_libraries(halo_gpu PUBLIC halo_model ${HALO_AMDHIP})
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
add_executable(halo_disk_bench tools/disk_bench.cpp)
target_link_libraries(gguf_inspect PRIVATE halo_core)
add_executable(halo_requant tools/requant.cpp)
target_link_libraries(halo_requant PRIVATE halo_core halo_model)

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
    tests/core/lowbit_test.cpp
    tests/core/json_test.cpp
    tests/core/sampling_test.cpp)
  target_compile_definitions(core_tests PRIVATE
    HALO_CORE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/core/data")
  halo_test(text_tests
    tests/text/tokenizer_test.cpp
    tests/text/chat_template_test.cpp
    tests/text/tool_call_parser_test.cpp)
  halo_test(server_tests
    tests/server/openai_request_test.cpp
    tests/server/openai_response_test.cpp
    tests/server/http_test.cpp
    tests/server/openai_service_test.cpp
    tests/server/systemone_service_test.cpp
    tests/server/metrics_test.cpp)
  target_link_libraries(server_tests PRIVATE halo_server)
  halo_test(decide_tests
    tests/decide/systemone_test.cpp
    tests/decide/decider_test.cpp)
  target_compile_definitions(decide_tests PRIVATE
    HALO_DECIDE_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/decide/data")
  if(HALO_GPU)
    halo_test(gpu_tests tests/gpu/qwen35_gpu_test.cpp tests/gpu/quant_gpu_test.cpp tests/gpu/flash_next_gpu_test.cpp tests/gpu/select_gpu_test.cpp tests/gpu/sample_gpu_test.cpp
      tests/gpu/flash_next_generator_test.cpp)
    target_link_libraries(gpu_tests PRIVATE halo_gpu)
    target_link_libraries(decide_tests PRIVATE halo_gpu)
    add_executable(halo_step_stats tools/step_stats.cpp)
    target_link_libraries(halo_step_stats PRIVATE halo_gpu)
    add_executable(halo_sample_bench tools/sample_bench.cpp)
    target_link_libraries(halo_sample_bench PRIVATE halo_gpu)
    add_library(halo_server_engine STATIC src/server/flash_next_generator.cpp)
    target_link_libraries(halo_server_engine PUBLIC halo_server halo_gpu)
    add_executable(halo_gemm_bench tools/gemm_bench.cpp)
    target_link_libraries(halo_gemm_bench PRIVATE halo_gpu)
    add_executable(halo_serve tools/serve.cpp)
    target_link_libraries(halo_serve PRIVATE halo_server_engine)
    if(TARGET gpu_tests)
      target_link_libraries(gpu_tests PRIVATE halo_server_engine)
    endif()
    add_executable(halo_run tools/run.cpp)
    target_link_libraries(halo_run PRIVATE halo_gpu halo_text)
    add_executable(halo_score tools/score.cpp)
    target_link_libraries(halo_score PRIVATE halo_gpu halo_text)
    add_executable(halo_graph_bench tools/graph_bench.cpp)
    target_link_libraries(halo_graph_bench PRIVATE halo_gpu)
    add_executable(halo_gemv_bench tools/gemv_bench.cpp)
    target_link_libraries(halo_gemv_bench PRIVATE halo_gpu)
  endif()
  halo_test(model_tests
    tests/model/reference_hf_test.cpp
    tests/model/ngram_table_test.cpp
    tests/model/routing_test.cpp
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

# --- scope: embedded LLM observability (P12 "halo-scope") --------------------
# Optional. libchdb is dlopen()ed at run time, never linked: point
# HALO_SCOPE_LIBCHDB (this cache path, or the environment variable of the
# same name at run time) at a libchdb.so. The documented default is
# chdb-core v26.9.0 (ClickHouse 26.9.2.8-stable), the release asset
# linux-x86_64-libchdb.tar.gz with
# sha256 c6398bcc71dc58d12fb81548540aacd5d8248830ec81cec537f51c114670543b,
# from github.com/chdb-io/chdb-core/releases. 26.7.0 also passes the tests.
option(HALO_SCOPE "Build halo-scope (embedded LLM observability; loads libchdb at run time)" OFF)
set(HALO_SCOPE_LIBCHDB "" CACHE FILEPATH "Default libchdb.so for halo-scope; $HALO_SCOPE_LIBCHDB overrides it")
if(HALO_SCOPE)
  include(FetchContent)
  find_package(SQLite3 QUIET)
  if(NOT SQLite3_FOUND)
    # No system sqlite3 headers: build the amalgamation.
    FetchContent_Declare(sqlite_amalgamation
      URL https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip
      URL_HASH SHA256=1e71ddf93849c6a6ecf58b827c0692073d2dd7ee40196158068f7b29f422e87d
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(sqlite_amalgamation)
    add_library(halo_sqlite3 STATIC ${sqlite_amalgamation_SOURCE_DIR}/sqlite3.c)
    target_include_directories(halo_sqlite3 SYSTEM PUBLIC ${sqlite_amalgamation_SOURCE_DIR})
    target_compile_definitions(halo_sqlite3 PRIVATE
      SQLITE_THREADSAFE=1 SQLITE_DQS=0 SQLITE_OMIT_LOAD_EXTENSION)
    target_compile_options(halo_sqlite3 PRIVATE -w)
    target_link_libraries(halo_sqlite3 PUBLIC Threads::Threads)
    add_library(SQLite::SQLite3 ALIAS halo_sqlite3)
  endif()

  # Payload content addressing: BLAKE3 (CC0 / Apache-2.0), fetched at build time.
  FetchContent_Declare(blake3
    URL https://github.com/BLAKE3-team/BLAKE3/archive/refs/tags/1.8.7.tar.gz
    URL_HASH SHA256=c6782a28842b1c0478524ac06a4f2ede784038ee298d6e2162c0b089c4306a3c
    SOURCE_SUBDIR c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  FetchContent_MakeAvailable(blake3)

  # HALO_SCOPE_MUTANT builds deliberately broken copies for the positive
  # controls below; 0 is the real library.
  function(halo_scope_library name mutant)
    add_library(${name} STATIC
      src/scope/chdb.cpp
      src/scope/migrate.cpp
      src/scope/schema.cpp
      src/scope/records.cpp
      src/scope/recorder.cpp
      src/scope/metadata.cpp
      src/scope/sqlite_util.cpp
      src/scope/evals_schema.cpp
      src/scope/wal.cpp
      src/scope/annotate.cpp
      src/scope/c_api.cpp)
    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${name} PUBLIC Threads::Threads ${CMAKE_DL_LIBS} SQLite::SQLite3)
    target_link_libraries(${name} PRIVATE BLAKE3::blake3)
    target_compile_definitions(${name} PRIVATE HALO_SCOPE_MUTANT=${mutant})
    if(HALO_SCOPE_LIBCHDB)
      target_compile_definitions(${name} PRIVATE HALO_SCOPE_LIBCHDB_DEFAULT="${HALO_SCOPE_LIBCHDB}")
    endif()
  endfunction()
  halo_scope_library(halo_scope 0)
  # Datasets, experiments and evaluators (P12, second half): SQLite ledger,
  # runner, evaluators and their C ABI.
  function(halo_scope_evals_library name scope_lib mutant)
    add_library(${name} STATIC
      src/scope/evals.cpp
      src/scope/eval_stats.cpp
      src/scope/json_schema.cpp
      src/scope/evaluators.cpp
      src/scope/http_client.cpp
      src/scope/runner.cpp
      src/scope/eval_json.cpp
      src/scope/c_api_evals.cpp)
    target_link_libraries(${name} PUBLIC ${scope_lib} halo_core)
    target_compile_definitions(${name} PRIVATE HALO_SCOPE_MUTANT=${mutant})
  endfunction()
  halo_scope_evals_library(halo_scope_evals halo_scope 0)
  # The OpenAI service's generations as observations (links the server).
  add_library(halo_scope_server STATIC src/scope/generation_observer.cpp src/scope/response_store.cpp
    src/scope/http_annotations.cpp src/scope/trace_reader.cpp src/scope/prompt_routes.cpp
    src/scope/eval_routes.cpp)
  target_link_libraries(halo_scope_server PUBLIC halo_scope halo_scope_evals halo_server)
  if(TARGET halo_serve)
    target_link_libraries(halo_serve PRIVATE halo_scope_server)
    target_compile_definitions(halo_serve PRIVATE HALO_WITH_SCOPE=1)
  endif()

  add_executable(scope_bench bench/scope/scope_bench.cpp)
  target_link_libraries(scope_bench PRIVATE halo_scope)
  # halo_scope_eval: datasets, evaluators and experiments from the shell,
  # over a halo_serve's /v1/scope API.
  add_executable(halo_scope_eval tools/scope_eval.cpp)
  target_link_libraries(halo_scope_eval PRIVATE halo_scope_evals)

  if(HALO_BUILD_TESTS)
    halo_test(scope_tests
      tests/scope/records_test.cpp
      tests/scope/recorder_test.cpp
      tests/scope/metadata_test.cpp
      tests/scope/wal_test.cpp
      tests/scope/annotate_test.cpp
      tests/scope/generation_observer_test.cpp
      tests/scope/response_store_test.cpp
      tests/scope/http_annotations_test.cpp
      tests/scope/trace_reader_test.cpp
      tests/scope/prompt_routes_test.cpp
      tests/scope/evals_test.cpp
      tests/scope/eval_stats_test.cpp
      tests/scope/evaluators_test.cpp
      tests/scope/runner_test.cpp
      tests/scope/eval_routes_test.cpp
      tests/scope/scope_main.cpp)
    target_link_libraries(scope_tests PRIVATE halo_scope halo_scope_server)

    # Positive controls: each mutant must make its property fail with the
    # expected message, so a crash, a skip (no libchdb) or an unrelated
    # failure does not count as caught.
    set(_scope_mutant_filter_1 ScopeRecords.ObservationRoundTripsThroughChdb)
    set(_scope_mutant_filter_2 ScopeRecords.ObservationRoundTripsThroughChdb)
    set(_scope_mutant_filter_3 ScopeWritePath.AccountingHoldsWithMemorySink)
    set(_scope_mutant_filter_4 ScopeWritePath.AccountingHoldsWithMemorySink)
    # The failure each mutant must produce, not merely "a failure".
    set(_scope_mutant_expect_1 "differs after the chDB round trip")
    set(_scope_mutant_expect_2 "differs after the chDB round trip")
    set(_scope_mutant_expect_3 "read back [0-9]+ \\+ refused [0-9]+ \\+ deduplicated [0-9]+ \\+ dropped [0-9]+ != produced")
    # a duplicated observation is a duplicate; a duplicated payload breaks the count
    set(_scope_mutant_expect_4 "duplicate row|read back [0-9]+ \\+ refused [0-9]+ \\+ deduplicated [0-9]+ \\+ dropped [0-9]+ != produced")
    set(_scope_mutant_filter_6 ScopeWal.ReplayIsIdempotentByWalPosition)
    set(_scope_mutant_expect_6 "duplicate rows after replay")
    set(_scope_mutant_filter_7 ScopeWritePath.AccountingHoldsWithMemorySink)
    set(_scope_mutant_expect_7 "suppressed but never written")
    set(_scope_mutants 1 2 3 4 6 7)
    if(HALO_SANITIZE MATCHES "thread")
      set(_scope_mutants 3 4 7)  # 1, 2 and 6 need chDB, which TSan builds do not load
    endif()
    foreach(m ${_scope_mutants})
      halo_scope_library(halo_scope_mutant${m} ${m})
      add_executable(scope_mutant${m}_tests
        tests/scope/records_test.cpp tests/scope/recorder_test.cpp tests/scope/wal_test.cpp
        tests/scope/scope_main.cpp)
      target_link_libraries(scope_mutant${m}_tests PRIVATE halo_scope_mutant${m} GTest::gtest hegel)
      target_compile_options(scope_mutant${m}_tests PRIVATE -Wno-missing-designated-field-initializers)
      add_test(NAME scope_mutant${m}_is_caught
        COMMAND scope_mutant${m}_tests --gtest_filter=${_scope_mutant_filter_${m}}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties(scope_mutant${m}_is_caught PROPERTIES
        ENVIRONMENT "HALO_SCOPE_TEST_WRITER=manual;HALO_SCOPE_TEST_CASES=2000"
        PASS_REGULAR_EXPRESSION "${_scope_mutant_expect_${m}}")
    endforeach()
    # Evaluation positive controls: 11 redoes done items on resume (and
    # drops the trigger that would refuse it), 12 lets frozen items change,
    # 13 computes the SE with n instead of n - 1, 14 mirrors scores after the
    # commit instead of before.
    set(_evals_mutant_filter_11 ScopeRunner.CrashesAnywhereResumeToTheUninterruptedResult)
    set(_evals_mutant_expect_11 "final resume|differs|counted twice|did work again")
    set(_evals_mutant_filter_12 ScopeEvals.FrozenVersionsAreImmutable)
    set(_evals_mutant_expect_12 "tamper accepted")
    set(_evals_mutant_filter_13 ScopeEvalStats.PairedComparisonMatchesReference)
    set(_evals_mutant_expect_13 "differs from the reference")
    set(_evals_mutant_filter_14 ScopeRunner.CrashesAnywhereResumeToTheUninterruptedResult)
    set(_evals_mutant_expect_14 "never mirrored")
    foreach(m 11 12 13 14)
      if(m EQUAL 11 OR m EQUAL 12)
        halo_scope_library(halo_scope_mutant${m} ${m})  # the schema lives in halo_scope
        halo_scope_evals_library(halo_scope_evals_mutant${m} halo_scope_mutant${m} ${m})
      else()
        halo_scope_evals_library(halo_scope_evals_mutant${m} halo_scope ${m})
      endif()
      add_executable(scope_evals_mutant${m}_tests
        tests/scope/evals_test.cpp tests/scope/eval_stats_test.cpp tests/scope/runner_test.cpp
        tests/scope/scope_main.cpp)
      target_link_libraries(scope_evals_mutant${m}_tests PRIVATE halo_scope_evals_mutant${m} GTest::gtest hegel)
      target_compile_options(scope_evals_mutant${m}_tests PRIVATE -Wno-missing-designated-field-initializers)
      add_test(NAME scope_evals_mutant${m}_is_caught
        COMMAND scope_evals_mutant${m}_tests --gtest_filter=${_evals_mutant_filter_${m}}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      # Serial and with a larger budget: two mutants share a property (and
      # its Hegel example database), and each must find its own failure.
      set_tests_properties(scope_evals_mutant${m}_is_caught PROPERTIES
        ENVIRONMENT "HALO_SCOPE_TEST_CASES=400"
        RUN_SERIAL TRUE
        PASS_REGULAR_EXPRESSION "${_evals_mutant_expect_${m}}")
    endforeach()
    if(HALO_SANITIZE MATCHES "thread")
      # TSan's own positive control: an unpublished ring write must be seen.
      halo_scope_library(halo_scope_mutant5 5)
      add_executable(scope_mutant5_tests tests/scope/recorder_test.cpp tests/scope/scope_main.cpp)
      target_link_libraries(scope_mutant5_tests PRIVATE halo_scope_mutant5 GTest::gtest hegel)
      target_compile_options(scope_mutant5_tests PRIVATE -Wno-missing-designated-field-initializers)
      add_test(NAME scope_mutant5_is_caught_by_tsan
        COMMAND scope_mutant5_tests --gtest_filter=ScopeWritePath.AccountingHoldsWithMemorySink
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
      set_tests_properties(scope_mutant5_is_caught_by_tsan PROPERTIES
        ENVIRONMENT "HALO_SCOPE_TEST_WRITER=thread;TSAN_OPTIONS=halt_on_error=1"
        PASS_REGULAR_EXPRESSION "WARNING: ThreadSanitizer: data race")
    endif()
  endif()
endif()

# Kernel register/scratch budget: fails on a new spill or scratch use (and on
# a stale allowlist entry). See tools/kernel_report.py.
if(TARGET halo_gpu)
  get_filename_component(_halo_hipcc_dir "${HIPCC}" DIRECTORY)
  get_filename_component(_halo_hipcc_dir "${_halo_hipcc_dir}" REALPATH)
  # The readelf of the LLVM that hipcc uses (distro: llvm-readelf-<major>).
  find_program(HALO_LLVM_READELF NAMES llvm-readelf llvm-readelf-21 llvm-readelf-20 llvm-readelf-19
               HINTS "${_halo_hipcc_dir}/../llvm/bin" "${_halo_hipcc_dir}/../lib/llvm/bin")
  find_package(Python3 COMPONENTS Interpreter)
  if(HALO_LLVM_READELF AND Python3_Interpreter_FOUND)
    add_test(NAME kernel_resources
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/kernel_report.py
                     --hsaco ${HALO_GEN}/core.hsaco --readelf ${HALO_LLVM_READELF}
                     --check ${CMAKE_CURRENT_SOURCE_DIR}/tools/kernel_spill_allowlist.txt)
    # The occupancy estimate against the compiler's own (recompiles core.hip).
    add_test(NAME kernel_occupancy_model
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/kernel_report.py
                     --compile ${HIPCC} --source ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/core.hip
                     --include ${CMAKE_CURRENT_SOURCE_DIR}/src --readelf ${HALO_LLVM_READELF})
    if(DEFINED ENV{HIP_DEVICE_LIB_PATH})  # hipcc needs it outside the dev shell too
      set_tests_properties(kernel_occupancy_model PROPERTIES
                           ENVIRONMENT "HIP_DEVICE_LIB_PATH=$ENV{HIP_DEVICE_LIB_PATH}")
    endif()
  endif()
endif()
