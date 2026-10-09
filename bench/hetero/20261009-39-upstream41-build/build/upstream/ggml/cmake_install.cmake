# Install script for directory: E:/Strata-Hetero-data/source/llama-3cf0325/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/strata")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("E:/Strata-Hetero-data/build/cuda-upstream41-03/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "E:/Strata-Hetero-data/build/cuda-upstream41-03/ggml/src/ggml.lib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-cpu.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-alloc.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-backend.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-blas.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-cann.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-cpp.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-cuda.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-opt.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-metal.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-rpc.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-virtgpu.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-sycl.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-vulkan.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-webgpu.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-zendnn.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/ggml-openvino.h"
    "E:/Strata-Hetero-data/source/llama-3cf0325/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "E:/Strata-Hetero-data/build/cuda-upstream41-03/ggml/src/ggml-base.lib")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "E:/Strata-Hetero-data/build/cuda-upstream41-03/ggml/ggml-config.cmake"
    "E:/Strata-Hetero-data/build/cuda-upstream41-03/ggml/ggml-config-version.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "E:/Strata-Hetero-data/build/cuda-upstream41-03/ggml/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
