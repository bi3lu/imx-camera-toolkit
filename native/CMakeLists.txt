cmake_minimum_required(VERSION 3.22)
project(imx_camera_cuda_interop LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87 CACHE STRING "CUDA architectures to compile")
option(IMX_CAMERA_ENABLE_HARDENING "Enable release compiler/linker hardening" ON)
option(IMX_CAMERA_ENABLE_SANITIZERS "Enable host ASan and UBSan instrumentation" OFF)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)

find_path(
    NVBUFSURFACE_INCLUDE_DIR
    nvbufsurface.h
    PATHS /usr/src/jetson_multimedia_api/include
    REQUIRED
)
find_path(
    PYGOBJECT_INCLUDE_DIR
    pygobject.h
    PATH_SUFFIXES pygobject-3.0
    DOC "PyGObject C API headers supplied by python-gi-dev"
    REQUIRED
)
find_library(
    NVBUFSURFACE_LIBRARY
    nvbufsurface
    PATHS /usr/lib/aarch64-linux-gnu/nvidia
    REQUIRED
)

pybind11_add_module(_cuda_interop MODULE src/cuda_interop.cu)
target_include_directories(
    _cuda_interop
    PRIVATE
    ${NVBUFSURFACE_INCLUDE_DIR}
    ${PYGOBJECT_INCLUDE_DIR}
)
target_link_libraries(
    _cuda_interop
    PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    PkgConfig::GSTREAMER
    ${NVBUFSURFACE_LIBRARY}
)

if(IMX_CAMERA_ENABLE_HARDENING)
    target_compile_options(
        _cuda_interop
        PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-fstack-protector-strong;-U_FORTIFY_SOURCE;-D_FORTIFY_SOURCE=3>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=-fstack-protector-strong;-U_FORTIFY_SOURCE;-D_FORTIFY_SOURCE=3>
    )
    target_link_options(
        _cuda_interop
        PRIVATE
        $<$<CONFIG:Release>:LINKER:SHELL:-z relro -z now>
    )
endif()

if(IMX_CAMERA_ENABLE_SANITIZERS)
    target_compile_options(
        _cuda_interop
        PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address;-fsanitize=undefined;-fno-omit-frame-pointer>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address;-Xcompiler=-fsanitize=undefined;-Xcompiler=-fno-omit-frame-pointer>
    )
    target_link_options(
        _cuda_interop
        PRIVATE
        -fsanitize=address
        -fsanitize=undefined
    )
endif()

if(DEFINED IMX_CAMERA_INTEROP_OUTPUT_DIRECTORY)
    set_target_properties(
        _cuda_interop
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${IMX_CAMERA_INTEROP_OUTPUT_DIRECTORY}"
    )
endif()
