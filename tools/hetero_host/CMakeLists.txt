cmake_minimum_required(VERSION 3.24)
project(strata_hetero_host LANGUAGES CXX)

# These upstream host paths need neither the CUDA SDK nor a GPU. Keep their
# fixture gate available before the much more expensive full engine build/run.
get_filename_component(_default_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(STRATA_SOURCE_ROOT "${_default_root}" CACHE PATH "Strata tree whose host code is tested")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)

add_library(strata_hetero_host STATIC
    "${STRATA_SOURCE_ROOT}/src/kernels/ngram.cpp"
    "${STRATA_SOURCE_ROOT}/src/ngram/ple_reader.cpp"
    "${STRATA_SOURCE_ROOT}/src/platform/direct_file.cpp"
    "${STRATA_SOURCE_ROOT}/src/platform/memory.cpp")
target_include_directories(strata_hetero_host PUBLIC "${STRATA_SOURCE_ROOT}/include")
target_link_libraries(strata_hetero_host PUBLIC Threads::Threads)
if(MSVC)
    target_compile_options(strata_hetero_host PRIVATE /W4 /utf-8)
else()
    target_compile_options(strata_hetero_host PRIVATE -Wall -Wextra)
endif()

add_executable(strata-ple-reader "${STRATA_SOURCE_ROOT}/src/ngram/ple_reader_test.cpp")
target_link_libraries(strata-ple-reader PRIVATE strata_hetero_host)
add_executable(strata-memory-check "${STRATA_SOURCE_ROOT}/src/platform/memory_test.cpp")
target_link_libraries(strata-memory-check PRIVATE strata_hetero_host)

enable_testing()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
add_test(NAME ple_reader_cpu COMMAND strata-ple-reader --selftest
    --dir "${CMAKE_CURRENT_BINARY_DIR}/fixtures")
add_test(NAME resident_memory_cpu COMMAND strata-memory-check)
