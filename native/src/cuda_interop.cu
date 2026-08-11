#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>

// CUDA defines ``__noinline__`` as a function qualifier while GLib probes it
// as a compiler attribute. Undefining it here avoids an NVCC/GLib preprocessor
// collision; this translation unit does not use the qualifier directly.
#ifdef __noinline__
#undef __noinline__
#endif

#include <gst/gst.h>
#include <nvbufsurface.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pygobject.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void check_driver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) {
        return;
    }

    const char* description = nullptr;
    cuGetErrorString(result, &description);
    throw std::runtime_error(
        std::string(operation) + ": " +
        (description == nullptr ? "unknown CUDA driver error" : description));
}

class CudaStream {
public:
    CudaStream() {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    std::uintptr_t handle() const {
        return reinterpret_cast<std::uintptr_t>(stream_);
    }

    cudaStream_t get() const { return stream_; }

    void synchronize() const {
        check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

private:
    cudaStream_t stream_{nullptr};
};

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t size) : size_(size) {
        if (size == 0) {
            throw std::invalid_argument("device buffer size must be positive");
        }
        check_cuda(cudaMalloc(&pointer_, size_), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    std::uintptr_t pointer() const {
        return reinterpret_cast<std::uintptr_t>(pointer_);
    }

    void* get() const { return pointer_; }
    std::size_t size() const { return size_; }

    py::bytes copy_to_host(const CudaStream& stream) const {
        std::string result(size_, '\0');
        check_cuda(
            cudaMemcpyAsync(
                result.data(),
                pointer_,
                size_,
                cudaMemcpyDeviceToHost,
                stream.get()),
            "cudaMemcpyAsync device-to-host");
        stream.synchronize();
        return py::bytes(result);
    }

private:
    void* pointer_{nullptr};
    std::size_t size_{0};
};

class NvmmSurface {
public:
    NvmmSurface(
        const py::object& gst_buffer,
        unsigned int expected_width,
        unsigned int expected_height) {
        if (!pyg_boxed_check(gst_buffer.ptr(), GST_TYPE_BUFFER)) {
            throw std::invalid_argument("object is not a boxed Gst.Buffer");
        }

        buffer_ = pyg_boxed_get(gst_buffer.ptr(), GstBuffer);
        if (buffer_ == nullptr) {
            throw std::invalid_argument("Gst.Buffer contains a null native pointer");
        }
        gst_buffer_ref(buffer_);

        if (!gst_buffer_map(buffer_, &map_info_, GST_MAP_READ)) {
            cleanup();
            throw std::runtime_error("could not inspect the NVMM Gst.Buffer");
        }

        mapped_ = true;
        surface_ = reinterpret_cast<NvBufSurface*>(map_info_.data);

        if (surface_ == nullptr || surface_->numFilled < 1 ||
            surface_->surfaceList == nullptr) {
            cleanup();
            throw std::runtime_error("Gst.Buffer does not contain NvBufSurface");
        }

        if (surface_->memType != NVBUF_MEM_SURFACE_ARRAY) {
            cleanup();
            throw std::runtime_error("NvBufSurface is not SURFACE_ARRAY memory");
        }

        NvBufSurfaceParams& parameters = surface_->surfaceList[0];

        if (parameters.width != expected_width ||
            parameters.height != expected_height) {
            cleanup();
            throw std::runtime_error("NvBufSurface dimensions do not match GpuFrame");
        }

        if (parameters.colorFormat != NVBUF_COLOR_FORMAT_NV12 &&
            parameters.colorFormat != NVBUF_COLOR_FORMAT_NV12_ER &&
            parameters.colorFormat != NVBUF_COLOR_FORMAT_NV12_709 &&
            parameters.colorFormat != NVBUF_COLOR_FORMAT_NV12_709_ER) {
            cleanup();
            throw std::runtime_error("NvBufSurface is not an NV12 format");
        }
        if (parameters.colorFormat == NVBUF_COLOR_FORMAT_NV12_ER) {
            color_matrix_ = 1;
        } else if (parameters.colorFormat == NVBUF_COLOR_FORMAT_NV12_709) {
            color_matrix_ = 2;
        } else if (parameters.colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER) {
            color_matrix_ = 3;
        }

        if (NvBufSurfaceMapEglImage(surface_, 0) != 0) {
            cleanup();
            throw std::runtime_error("NvBufSurfaceMapEglImage failed");
        }

        egl_mapped_ = true;

        auto image = static_cast<EGLImageKHR>(parameters.mappedAddr.eglImage);

        if (image == nullptr) {
            cleanup();
            throw std::runtime_error("NvBufSurface returned no EGLImage");
        }

        try {
            check_driver(
                cuGraphicsEGLRegisterImage(
                    &cuda_resource_,
                    image,
                    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE),
                "cuGraphicsEGLRegisterImage");
            cuda_registered_ = true;
            check_driver(
                cuGraphicsResourceGetMappedEglFrame(
                    &egl_frame_, cuda_resource_, 0, 0),
                "cuGraphicsResourceGetMappedEglFrame");
        } catch (...) {
            cleanup();
            throw;
        }

        if (egl_frame_.planeCount < 2) {
            cleanup();
            throw std::runtime_error("CUDA EGL frame has fewer than two NV12 planes");
        }

        width_ = parameters.width;
        height_ = parameters.height;
        y_pitch_ = parameters.planeParams.pitch[0];
        uv_pitch_ = parameters.planeParams.pitch[1];

        if (y_pitch_ == 0) {
            y_pitch_ = egl_frame_.pitch;
        }

        if (uv_pitch_ == 0) {
            uv_pitch_ = y_pitch_;
        }

        if (egl_frame_.frameType == CU_EGL_FRAME_TYPE_ARRAY) {
            cudaResourceDesc y_description{};
            y_description.resType = cudaResourceTypeArray;
            y_description.res.array.array =
                reinterpret_cast<cudaArray_t>(egl_frame_.frame.pArray[0]);
            check_cuda(
                cudaCreateSurfaceObject(&y_surface_, &y_description),
                "cudaCreateSurfaceObject Y");

            cudaResourceDesc uv_description{};
            uv_description.resType = cudaResourceTypeArray;
            uv_description.res.array.array =
                reinterpret_cast<cudaArray_t>(egl_frame_.frame.pArray[1]);

            try {
                check_cuda(
                    cudaCreateSurfaceObject(&uv_surface_, &uv_description),
                    "cudaCreateSurfaceObject UV");
            } catch (...) {
                cleanup();
                throw;
            }
        } else if (egl_frame_.frameType != CU_EGL_FRAME_TYPE_PITCH) {
            cleanup();
            throw std::runtime_error("unsupported CUDA EGL frame type");
        }
    }

    ~NvmmSurface() { cleanup(); }

    NvmmSurface(const NvmmSurface&) = delete;
    NvmmSurface& operator=(const NvmmSurface&) = delete;

    const CUeglFrame& frame() const { return egl_frame_; }
    unsigned int width() const { return width_; }
    unsigned int height() const { return height_; }
    unsigned int y_pitch() const { return y_pitch_; }
    unsigned int uv_pitch() const { return uv_pitch_; }
    cudaSurfaceObject_t y_surface() const { return y_surface_; }
    cudaSurfaceObject_t uv_surface() const { return uv_surface_; }
    int color_matrix() const { return color_matrix_; }

private:
    void cleanup() noexcept {
        if (uv_surface_ != 0) {
            cudaDestroySurfaceObject(uv_surface_);
            uv_surface_ = 0;
        }

        if (y_surface_ != 0) {
            cudaDestroySurfaceObject(y_surface_);
            y_surface_ = 0;
        }

        if (cuda_registered_) {
            cuGraphicsUnregisterResource(cuda_resource_);
            cuda_registered_ = false;
        }

        if (egl_mapped_ && surface_ != nullptr) {
            NvBufSurfaceUnMapEglImage(surface_, 0);
            egl_mapped_ = false;
        }

        if (mapped_ && buffer_ != nullptr) {
            gst_buffer_unmap(buffer_, &map_info_);
            mapped_ = false;
        }

        if (buffer_ != nullptr) {
            gst_buffer_unref(buffer_);
            buffer_ = nullptr;
        }
    }

    GstBuffer* buffer_{nullptr};
    GstMapInfo map_info_ = GST_MAP_INFO_INIT;
    NvBufSurface* surface_{nullptr};
    CUgraphicsResource cuda_resource_{nullptr};
    CUeglFrame egl_frame_{};
    bool mapped_{false};
    bool egl_mapped_{false};
    bool cuda_registered_{false};
    unsigned int width_{0};
    unsigned int height_{0};
    unsigned int y_pitch_{0};
    unsigned int uv_pitch_{0};
    cudaSurfaceObject_t y_surface_{0};
    cudaSurfaceObject_t uv_surface_{0};
    int color_matrix_{0};
};

__device__ unsigned char clamp_byte(float value) {
    return static_cast<unsigned char>(fminf(fmaxf(value, 0.0F), 255.0F));
}

__device__ unsigned char read_array_byte(
    cudaSurfaceObject_t surface,
    int byte_x,
    int y) {
    unsigned char value = 0;
    surf2Dread(&value, surface, byte_x, y);
    return value;
}

__device__ void write_array_byte(
    cudaSurfaceObject_t surface,
    int byte_x,
    int y,
    unsigned char value) {
    surf2Dwrite(value, surface, byte_x, y);
}

__global__ void draw_nv12_rectangle_kernel(
    unsigned char* y_plane,
    unsigned char* uv_plane,
    unsigned int y_pitch,
    unsigned int uv_pitch,
    cudaSurfaceObject_t y_surface,
    cudaSurfaceObject_t uv_surface,
    bool array_frame,
    unsigned int surface_width,
    unsigned int left,
    unsigned int top,
    unsigned int rectangle_width,
    unsigned int rectangle_height,
    unsigned int thickness,
    unsigned char y_value,
    unsigned char u_value,
    unsigned char v_value) {
    const unsigned int local_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_x >= rectangle_width || local_y >= rectangle_height) {
        return;
    }

    const bool border =
        local_x < thickness || local_y < thickness ||
        local_x + thickness >= rectangle_width ||
        local_y + thickness >= rectangle_height;

    if (!border) {
        return;
    }
    const unsigned int x = left + local_x;
    const unsigned int y = top + local_y;

    if (array_frame) {
        write_array_byte(y_surface, static_cast<int>(x), y, y_value);
    } else {
        y_plane[y * y_pitch + x] = y_value;
    }

    if ((x & 1U) != 0 || (y & 1U) != 0 || x + 1 >= surface_width) {
        return;
    }

    const unsigned int uv_y = y / 2;

    if (array_frame) {
        write_array_byte(uv_surface, static_cast<int>(x), uv_y, u_value);
        write_array_byte(uv_surface, static_cast<int>(x + 1), uv_y, v_value);
    } else {
        uv_plane[uv_y * uv_pitch + x] = u_value;
        uv_plane[uv_y * uv_pitch + x + 1] = v_value;
    }
}

__global__ void nv12_to_nchw_kernel(
    const unsigned char* y_plane,
    const unsigned char* uv_plane,
    unsigned int y_pitch,
    unsigned int uv_pitch,
    cudaSurfaceObject_t y_surface,
    cudaSurfaceObject_t uv_surface,
    bool array_frame,
    unsigned int source_width,
    unsigned int source_height,
    int color_matrix,
    float* destination,
    unsigned int output_width,
    unsigned int output_height,
    unsigned int resized_width,
    unsigned int resized_height,
    unsigned int pad_x,
    unsigned int pad_y,
    bool rgb,
    float scale,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2,
    float padding_red,
    float padding_green,
    float padding_blue) {
    const unsigned int output_x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int output_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (output_x >= output_width || output_y >= output_height) {
        return;
    }

    const unsigned int plane_size = output_width * output_height;
    const unsigned int offset = output_y * output_width + output_x;
    if (output_x < pad_x || output_y < pad_y ||
        output_x >= pad_x + resized_width ||
        output_y >= pad_y + resized_height) {
        const float channel0 = rgb ? padding_red : padding_blue;
        const float channel2 = rgb ? padding_blue : padding_red;
        destination[offset] = (channel0 * scale - mean0) / std0;
        destination[plane_size + offset] =
            (padding_green * scale - mean1) / std1;
        destination[2 * plane_size + offset] =
            (channel2 * scale - mean2) / std2;
        return;
    }

    const unsigned int resized_x = output_x - pad_x;
    const unsigned int resized_y = output_y - pad_y;

    const unsigned int source_x = min(
        static_cast<unsigned int>(
            (static_cast<unsigned long long>(resized_x) * source_width) /
            resized_width),
        source_width - 1);

    const unsigned int source_y = min(
        static_cast<unsigned int>(
            (static_cast<unsigned long long>(resized_y) * source_height) /
            resized_height),
        source_height - 1);

    const unsigned int uv_x = source_x & ~1U;
    const unsigned int uv_y = source_y / 2;

    unsigned char y_value;
    unsigned char u_value;
    unsigned char v_value;

    if (array_frame) {
        y_value = read_array_byte(y_surface, static_cast<int>(source_x), source_y);
        u_value = read_array_byte(uv_surface, static_cast<int>(uv_x), uv_y);
        v_value = read_array_byte(uv_surface, static_cast<int>(uv_x + 1), uv_y);
    } else {
        y_value = y_plane[source_y * y_pitch + source_x];
        u_value = uv_plane[uv_y * uv_pitch + uv_x];
        v_value = uv_plane[uv_y * uv_pitch + uv_x + 1];
    }

    float y;
    float red_coefficient;
    float green_u_coefficient;
    float green_v_coefficient;
    float blue_coefficient;
    if (color_matrix == 1) {
        y = static_cast<float>(y_value);
        red_coefficient = 1.402F;
        green_u_coefficient = 0.344F;
        green_v_coefficient = 0.714F;
        blue_coefficient = 1.772F;
    } else if (color_matrix == 2) {
        y = 1.164F * (static_cast<float>(y_value) - 16.0F);
        red_coefficient = 1.793F;
        green_u_coefficient = 0.213F;
        green_v_coefficient = 0.533F;
        blue_coefficient = 2.112F;
    } else if (color_matrix == 3) {
        y = static_cast<float>(y_value);
        red_coefficient = 1.575F;
        green_u_coefficient = 0.187F;
        green_v_coefficient = 0.468F;
        blue_coefficient = 1.856F;
    } else {
        y = 1.164F * (static_cast<float>(y_value) - 16.0F);
        red_coefficient = 1.596F;
        green_u_coefficient = 0.392F;
        green_v_coefficient = 0.813F;
        blue_coefficient = 2.017F;
    }
    const float u = static_cast<float>(u_value) - 128.0F;
    const float v = static_cast<float>(v_value) - 128.0F;
    const float red =
        static_cast<float>(clamp_byte(y + red_coefficient * v));
    const float green =
        static_cast<float>(clamp_byte(
            y - green_u_coefficient * u - green_v_coefficient * v));
    const float blue =
        static_cast<float>(clamp_byte(y + blue_coefficient * u));

    const float channel0 = rgb ? red : blue;
    const float channel2 = rgb ? blue : red;
    destination[offset] = (channel0 * scale - mean0) / std0;
    destination[plane_size + offset] = (green * scale - mean1) / std1;
    destination[2 * plane_size + offset] = (channel2 * scale - mean2) / std2;
}

void preprocess_nv12(
    const NvmmSurface& source,
    DeviceBuffer& destination,
    unsigned int output_width,
    unsigned int output_height,
    const std::string& channel_order,
    float scale,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& standard_deviation,
    const std::string& resize_mode,
    const std::array<float, 3>& padding_value,
    const CudaStream& stream) {
    if (output_width == 0 || output_height == 0) {
        throw std::invalid_argument("output dimensions must be positive");
    }

    if (channel_order != "RGB" && channel_order != "BGR") {
        throw std::invalid_argument("channel_order must be RGB or BGR");
    }

    if (resize_mode != "stretch" && resize_mode != "letterbox") {
        throw std::invalid_argument("resize_mode must be stretch or letterbox");
    }

    if (std::any_of(
            padding_value.begin(),
            padding_value.end(),
            [](float value) {
                return !std::isfinite(value) || value < 0.0F || value > 255.0F;
            })) {
        throw std::invalid_argument("padding values must be between 0 and 255");
    }

    if (std::any_of(
            standard_deviation.begin(),
            standard_deviation.end(),
            [](float value) { return value == 0.0F; })) {
        throw std::invalid_argument("standard deviation values must be non-zero");
    }

    const std::size_t required_size =
        static_cast<std::size_t>(3) * output_width * output_height * sizeof(float);

    if (destination.size() < required_size) {
        throw std::invalid_argument("destination device buffer is too small");
    }

    const CUeglFrame& frame = source.frame();
    const bool is_array = frame.frameType == CU_EGL_FRAME_TYPE_ARRAY;
    const auto* y_plane = is_array
        ? nullptr
        : static_cast<const unsigned char*>(frame.frame.pPitch[0]);
    const auto* uv_plane = is_array
        ? nullptr
        : static_cast<const unsigned char*>(frame.frame.pPitch[1]);

    unsigned int resized_width = output_width;
    unsigned int resized_height = output_height;
    unsigned int pad_x = 0;
    unsigned int pad_y = 0;
    if (resize_mode == "letterbox") {
        const double uniform_scale = std::min(
            static_cast<double>(output_width) / source.width(),
            static_cast<double>(output_height) / source.height());
        resized_width = std::min(
            output_width,
            std::max(1U, static_cast<unsigned int>(
                std::lround(source.width() * uniform_scale))));
        resized_height = std::min(
            output_height,
            std::max(1U, static_cast<unsigned int>(
                std::lround(source.height() * uniform_scale))));
        pad_x = (output_width - resized_width) / 2;
        pad_y = (output_height - resized_height) / 2;
    }

    const dim3 threads(16, 16);
    const dim3 blocks(
        (output_width + threads.x - 1) / threads.x,
        (output_height + threads.y - 1) / threads.y);
    nv12_to_nchw_kernel<<<blocks, threads, 0, stream.get()>>>(
        y_plane,
        uv_plane,
        source.y_pitch(),
        source.uv_pitch(),
        source.y_surface(),
        source.uv_surface(),
        is_array,
        source.width(),
        source.height(),
        source.color_matrix(),
        static_cast<float*>(destination.get()),
        output_width,
        output_height,
        resized_width,
        resized_height,
        pad_x,
        pad_y,
        channel_order == "RGB",
        scale,
        mean[0],
        mean[1],
        mean[2],
        standard_deviation[0],
        standard_deviation[1],
        standard_deviation[2],
        padding_value[0],
        padding_value[1],
        padding_value[2]);
    check_cuda(cudaGetLastError(), "NV12 preprocessing kernel launch");
}

void draw_nv12_rectangle(
    NvmmSurface& surface,
    unsigned int left,
    unsigned int top,
    unsigned int width,
    unsigned int height,
    unsigned int thickness,
    unsigned char y_value,
    unsigned char u_value,
    unsigned char v_value,
    const CudaStream& stream) {
    if (width == 0 || height == 0 || thickness == 0) {
        throw std::invalid_argument(
            "rectangle dimensions and thickness must be positive");
    }

    if (left >= surface.width() || top >= surface.height()) {
        return;
    }

    const unsigned int clipped_width =
        std::min(width, surface.width() - left);
    const unsigned int clipped_height =
        std::min(height, surface.height() - top);
    const unsigned int clipped_thickness =
        std::min(thickness, std::min(clipped_width, clipped_height));
    const CUeglFrame& frame = surface.frame();
    const bool is_array = frame.frameType == CU_EGL_FRAME_TYPE_ARRAY;
    auto* y_plane = is_array
        ? nullptr
        : static_cast<unsigned char*>(frame.frame.pPitch[0]);
    auto* uv_plane = is_array
        ? nullptr
        : static_cast<unsigned char*>(frame.frame.pPitch[1]);
    const dim3 threads(16, 16);
    const dim3 blocks(
        (clipped_width + threads.x - 1) / threads.x,
        (clipped_height + threads.y - 1) / threads.y);
    draw_nv12_rectangle_kernel<<<blocks, threads, 0, stream.get()>>>(
        y_plane,
        uv_plane,
        surface.y_pitch(),
        surface.uv_pitch(),
        surface.y_surface(),
        surface.uv_surface(),
        is_array,
        surface.width(),
        left,
        top,
        clipped_width,
        clipped_height,
        clipped_thickness,
        y_value,
        u_value,
        v_value);
    check_cuda(cudaGetLastError(), "NV12 rectangle kernel launch");
}

std::pair<int, int> compute_capability() {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    return {properties.major, properties.minor};
}

}  // namespace

PYBIND11_MODULE(_cuda_interop, module) {
    PyObject* pygobject = pygobject_init(-1, -1, -1);
    if (pygobject == nullptr) {
        throw py::error_already_set();
    }
    Py_DECREF(pygobject);

    module.doc() = "Jetson NvBufSurface/EGLImage CUDA interoperability";

    py::class_<CudaStream>(module, "CudaStream")
        .def(py::init<>())
        .def_property_readonly("handle", &CudaStream::handle)
        .def("synchronize", &CudaStream::synchronize);

    py::class_<DeviceBuffer>(module, "DeviceBuffer")
        .def(py::init<std::size_t>())
        .def_property_readonly("pointer", &DeviceBuffer::pointer)
        .def_property_readonly("size", &DeviceBuffer::size)
        .def("copy_to_host", &DeviceBuffer::copy_to_host);

    py::class_<NvmmSurface>(module, "NvmmSurface")
        .def(py::init<const py::object&, unsigned int, unsigned int>())
        .def_property_readonly("width", &NvmmSurface::width)
        .def_property_readonly("height", &NvmmSurface::height);

    module.def("compute_capability", &compute_capability);
    module.def(
        "preprocess_nv12",
        &preprocess_nv12,
        py::arg("source"),
        py::arg("destination"),
        py::arg("output_width"),
        py::arg("output_height"),
        py::arg("channel_order") = "RGB",
        py::arg("scale") = 1.0F / 255.0F,
        py::arg("mean") = std::array<float, 3>{0.0F, 0.0F, 0.0F},
        py::arg("standard_deviation") = std::array<float, 3>{1.0F, 1.0F, 1.0F},
        py::arg("resize_mode") = "stretch",
        py::arg("padding_value") = std::array<float, 3>{114.0F, 114.0F, 114.0F},
        py::arg("stream"));
    module.def(
        "draw_nv12_rectangle",
        &draw_nv12_rectangle,
        py::arg("surface"),
        py::arg("left"),
        py::arg("top"),
        py::arg("width"),
        py::arg("height"),
        py::arg("thickness"),
        py::arg("y"),
        py::arg("u"),
        py::arg("v"),
        py::arg("stream"));
}
