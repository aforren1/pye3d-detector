cmake_minimum_required(VERSION 3.21...4.0)

project("pye3d" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(APPLE)
    # We target older macOS where c++17 is spelled c++1z on AppleClang.
    # See https://clang.llvm.org/cxx_status.html
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
endif()

# scikit-build-core hands us a configured FindPython; cython-cmake provides the
# UseCython helpers (add it to build-system.requires). NumPy is intentionally
# not required: the .pyx files use typed memoryviews (buffer protocol), not the
# numpy C-API.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# scikit-build-core sets SKBUILD_SABI_VERSION (from wheel.py-api) only for abi3
# builds (CPython 3.11+ here); on 3.10 it is unset and we build a normal
# version-specific module. USE_SABI then defines Py_LIMITED_API + .abi3 suffix.
if(DEFINED SKBUILD_SABI_VERSION AND NOT SKBUILD_SABI_VERSION STREQUAL "")
    set(_sabi_args USE_SABI ${SKBUILD_SABI_VERSION})
else()
    set(_sabi_args)
endif()

# Each module is a C++ Cython extension. They cimport common_types.pxd (Eigen
# typedefs) and pull in sibling C++ headers (search_on_sphere.h, etc.), so every
# target gets the source dir on its include path and links Eigen.
set(PYE3D_MODULES pupil_detection_3d projections refraction_correction)
foreach(mod IN LISTS PYE3D_MODULES)
    cython_transpile("${mod}.pyx"
        LANGUAGE CXX
        CYTHON_ARGS -3 -I "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE ${mod}_cxx)
    python_add_library(${mod} MODULE WITH_SOABI ${_sabi_args} "${${mod}_cxx}")
    target_include_directories(${mod} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${mod} PRIVATE Eigen3::Eigen)
    install(TARGETS ${mod} DESTINATION "pye3d/cpp")
endforeach()
