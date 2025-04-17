cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# Toy train-inference engine
project(ttie VERSION 1.0 LANGUAGES CXX)

# ------------------------------------------- Library

add_library(ttie INTERFACE)
target_compile_features(ttie INTERFACE cxx_std_17)
target_include_directories(ttie INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# ------------------------------------------- Example

add_executable(example example/main.cpp)
target_link_libraries(example PRIVATE ttie)

# ------------------------------------------- Tests

enable_testing()

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP true
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(tests tests/test_main.cpp)
target_link_libraries(tests PRIVATE gtest gtest_main ttie)

add_test(NAME tests COMMAND tests)
