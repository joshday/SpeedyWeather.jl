module SpeedyWeatherInternalsAMDGPUExt

import AMDGPU: ROCArray, ROCDeviceArray, ROCBackend, Runtime
import SpeedyWeatherInternals.Architectures: Architectures, GPU, CPU, ROCGPU, array_type, architecture, on_architecture, compatible_array_types, nonparametric_type

const ROCGPU_Type = GPU{ROCBackend}

# DEVICE SETUP FOR AMDGPU
# extend functions from Architectures
Architectures.array_type(::ROCGPU_Type) = ROCArray
Architectures.array_type(::Type{<:ROCGPU_Type}) = ROCArray
Architectures.array_type(::ROCGPU_Type, NF::Type, N::Int) = ROCArray{NF, N, Runtime.Mem.HIPBuffer}

Architectures.compatible_array_types(::ROCGPU_Type) = (ROCArray, ROCDeviceArray)
Architectures.compatible_array_types(::Type{<:ROCGPU_Type}) = (ROCArray, ROCDeviceArray)

Architectures.nonparametric_type(::Type{<:ROCArray}) = ROCArray

Architectures.ROCGPU() = GPU(ROCBackend())
Architectures.GPU() = ROCGPU()

Architectures.architecture(::ROCArray) = ROCGPU()
Architectures.architecture(::Type{<:ROCArray}) = ROCGPU()
Architectures.architecture(::Type{<:ROCDeviceArray}) = ROCGPU()

Architectures.on_architecture(::CPU, a::ROCArray) = Array(a)
Architectures.on_architecture(::ROCGPU_Type, a::ROCArray) = a
Architectures.on_architecture(::CPU, a::SubArray{<:Any, <:Any, <:ROCArray}) = Array(a)

Architectures.on_architecture(::ROCGPU_Type, a::Array) = ROCArray(a)
Architectures.on_architecture(::ROCGPU_Type, a::BitArray) = ROCArray(a)
Architectures.on_architecture(::ROCGPU_Type, a::SubArray{<:Any, <:Any, <:ROCArray}) = a
Architectures.on_architecture(::ROCGPU_Type, a::SubArray{<:Any, <:Any, <:Array}) = ROCArray(a)
Architectures.on_architecture(::ROCGPU_Type, a::StepRangeLen) = a

end
