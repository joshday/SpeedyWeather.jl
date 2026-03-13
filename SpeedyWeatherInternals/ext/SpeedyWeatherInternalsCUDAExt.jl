module SpeedyWeatherInternalsCUDAExt

import CUDA: CUDA, CuArray, CuDeviceArray
import SpeedyWeatherInternals.Architectures: Architectures, GPU, CPU, CUDAGPU, array_type, architecture, on_architecture, compatible_array_types, nonparametric_type

const CuGPU = GPU{CUDA.CUDABackend{true}}

# DEVICE SETUP FOR CUDA
# extend functions from Architectures
Architectures.array_type(::CuGPU) = CuArray
Architectures.array_type(::Type{<:CuGPU}) = CuArray
Architectures.array_type(::CuGPU, NF::Type, N::Int) = CuArray{NF, N, CUDA.DeviceMemory}

Architectures.compatible_array_types(::CuGPU) = (CuArray, CuDeviceArray)
Architectures.compatible_array_types(::Type{<:CuGPU}) = (CuArray, CuDeviceArray)

Architectures.nonparametric_type(::Type{<:CuArray}) = CuArray

Architectures.CUDAGPU() = GPU(CUDA.CUDABackend(always_inline = true))
Architectures.GPU() = CUDAGPU() # default to CUDA

Architectures.architecture(::CuArray) = CUDAGPU()
Architectures.architecture(::Type{<:CuArray}) = CUDAGPU()
Architectures.architecture(::Type{<:CuDeviceArray}) = CUDAGPU()

Architectures.on_architecture(::CPU, a::CuArray) = Array(a)
Architectures.on_architecture(::CuGPU, a::CuArray) = a
Architectures.on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:CuArray}) = Array(a)

Architectures.on_architecture(::CuGPU, a::Array) = CuArray(a)
Architectures.on_architecture(::CuGPU, a::BitArray) = CuArray(a)
Architectures.on_architecture(::CuGPU, a::SubArray{<:Any, <:Any, <:CuArray}) = a
Architectures.on_architecture(::CuGPU, a::SubArray{<:Any, <:Any, <:Array}) = CuArray(a)
Architectures.on_architecture(::CuGPU, a::StepRangeLen) = a

end
