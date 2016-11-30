#ifndef MARVIN_HPP
#define MARVIN_HPP

/*
 * ----------------------------------------------------------------------------
 * Marvin: A Minimalist GPU-only N-Dimensional ConvNets Framework
 * Copyright (C) 2016 AutoX, Inc.
 * ----------------------------------------------------------------------------
 */

#if DATATYPE==0
    #pragma message "Compiling using StorageT=half ComputeT=float"
    #define StorageT half
    #define ComputeT float
    #define sizeofStorageT 2
    #define sizeofComputeT 4
    #define CUDNNStorageT CUDNN_DATA_HALF
    #define CUDNNConvComputeT CUDNN_DATA_FLOAT
    #define CPUStorage2ComputeT(x) (marvin::cpu_half2float(x))
    #define CPUCompute2StorageT(x) (marvin::cpu_float2half(x))
    #define GPUStorage2ComputeT(x) (__half2float(x))
    #define GPUCompute2StorageT(x) (__float2half(x))
    #define GPUgemm Hgemm
    #define GPUasum Hasum
    #define ISNAN(x) (ishnan(x))
    #define ComputeT_MIN FLT_MIN
    #include <cuda_fp16.h>
#elif DATATYPE==1
    #pragma message "Compiling using StorageT=float ComputeT=float"
    #define StorageT float
    #define ComputeT float
    #define sizeofStorageT 4
    #define sizeofComputeT 4
    #define CUDNNStorageT CUDNN_DATA_FLOAT
    #define CUDNNConvComputeT CUDNN_DATA_FLOAT
    #define CPUStorage2ComputeT(x) (x)
    #define CPUCompute2StorageT(x) (x)
    #define GPUStorage2ComputeT(x) (x)
    #define GPUCompute2StorageT(x) (x)
    #define GPUgemm cublasSgemm
    #define GPUasum cublasSasum
    #define ISNAN(x) (std::isnan(x))
    #define ComputeT_MIN FLT_MIN
#elif DATATYPE==2
    #pragma message "Compiling using StorageT=double ComputeT=double"
    #define StorageT double
    #define ComputeT double
    #define sizeofStorageT 8
    #define sizeofComputeT 8
    #define CUDNNStorageT CUDNN_DATA_DOUBLE
    #define CUDNNConvComputeT CUDNN_DATA_DOUBLE
    #define CPUStorage2ComputeT(x) (x)
    #define CPUCompute2StorageT(x) (x)
    #define GPUStorage2ComputeT(x) (x)
    #define GPUCompute2StorageT(x) (x)
    #define GPUgemm cublasDgemm
    #define GPUasum cublasDasum
    #define ISNAN(x) (std::isnan(x))
    #define ComputeT_MIN DBL_MIN
#endif

#if CUDA_VERSION >= 8000
#define CUBLAS_DATA_HALF CUDA_R_16F
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////
// Includes
//////////////////////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <typeinfo>
#include <typeindex>
#include <thread>
#include <chrono>
#include <future>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cudnn.h>
#include <sys/time.h>

#define USE_OPENCV 1

#if USE_OPENCV
// using namespace std;
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif

namespace marvin {

//////////////////////////////////////////////////////////////////////////////////////////////////
// Type definition
//////////////////////////////////////////////////////////////////////////////////////////////////

enum Filler { Xavier, Gaussian, Constant };
enum Pool { Max, Average, Sum };
enum LossObjective { MultinomialLogistic_StableSoftmax, MultinomialLogistic, SmoothL1, Contrastive, EuclideanSSE, HingeL1, HingeL2, SigmoidCrossEntropy, Infogain };
enum Phase { Training, Testing, TrainingTesting };
enum LRPolicy { LR_fixed, LR_step, LR_exp, LR_inv, LR_multistep, LR_poly, LR_sigmoid, LR_cyclical };
enum SolverAlgorithm { SGD, AdaDelta, AdaGrad, Adam, NAG, RMSprop};
enum Regularizer { L2, L1 };
enum LRN { CrossChannel, DivisiveNormalization };
enum ElementWiseOp { ElementWise_EQL, ElementWise_MUL, ElementWise_SUM, ElementWise_MIN, ElementWise_MAX };


ComputeT anyval;
ComputeT oneval = 1;
ComputeT zeroval = 0;
const void* one = static_cast<void *>(&oneval);
const void* zero = static_cast<void *>(&zeroval);
const ComputeT* oneComputeT = &oneval;
const ComputeT* zeroComputeT = &zeroval;

//////////////////////////////////////////////////////////////////////////////////////////////////
// Debugging utility
//////////////////////////////////////////////////////////////////////////////////////////////////

void FatalError(const int lineNumber=0) {
    std::cerr << "FatalError";
    if (lineNumber!=0) std::cerr<<" at LINE "<<lineNumber;
    std::cerr << ". Program Terminated." << std::endl;
    cudaDeviceReset();
    exit(EXIT_FAILURE);
}

void checkCUDA(const int lineNumber, cudaError_t error_code) {
    if (error_code != cudaSuccess) {
        std::cerr << "CUDA failure at LINE " << lineNumber << ": cudaError code " << error_code << ": ";

        std::string msg;

        switch(error_code){

            case cudaSuccess: //0    
                msg = "The API call returned with no errors. In the case of query calls, this can also mean that the operation being queried is complete (see ::cudaEventQuery() and ::cudaStreamQuery())."; break;

            case cudaErrorMissingConfiguration: //1
                msg = "The device function being invoked (usually via ::cudaLaunchKernel()) was not previously configured via the ::cudaConfigureCall() function."; break;

            case cudaErrorMemoryAllocation: //2
                msg = "The API call failed because it was unable to allocate enough memory to perform the requested operation."; break;
              
            case cudaErrorInitializationError: //3
                msg = "The API call failed because the CUDA driver and runtime could not be initialized."; break;

            case cudaErrorLaunchFailure: //4,
                msg = "An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointer and accessing out of bounds shared memory. The device cannot be used until ::cudaThreadExit() is called. All existing device memory allocations are invalid and must be reconstructed if the program is to continue using CUDA."; break;
              
            case cudaErrorPriorLaunchFailure  : //5,
                msg = "This indicated that a previous kernel launch failed. This was previously used for device emulation of kernel launches. [deprecated] This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release."; break;

            case cudaErrorLaunchTimeout: //6,
                msg = "This indicates that the device kernel took too long to execute. This can only occur if timeouts are enabled - see the device property \ref ::cudaDeviceProp::kernelExecTimeoutEnabled kernelExecTimeoutEnabled for more information. The device cannot be used until ::cudaThreadExit() is called. All existing device memory allocations are invalid and must be reconstructed if the program is to continue using CUDA."; break;
              
            case cudaErrorLaunchOutOfResources: //7,
                msg = "This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to ::cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."; break;
              
            case cudaErrorInvalidDeviceFunction    : //  8,
                msg = "The requested device function does not exist or is not compiled for the proper device architecture."; break;

            case cudaErrorInvalidConfiguration: //9, 
                msg = "This indicates that a kernel launch is requesting resources that can never be satisfied by the current device. Requesting more shared memory per block than the device supports will trigger this error, as will requesting too many threads or blocks. See ::cudaDeviceProp for more device limitations."; break;

            case cudaErrorInvalidDevice            : // 10,
                msg = "This indicates that the device ordinal supplied by the user does not correspond to a valid CUDA device."; break;

            case cudaErrorInvalidValue             : // 11,
                msg = "This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values."; break;

            case cudaErrorInvalidPitchValue        : // 12,
                msg = "This indicates that one or more of the pitch-related parameters passed to the API call is not within the acceptable range for pitch."; break;

            case cudaErrorInvalidSymbol            : // 13,
                msg = "This indicates that the symbol name/identifier passed to the API call is not a valid name or identifier."; break;

            case cudaErrorMapBufferObjectFailed    : // 14,
                msg = "This indicates that the buffer object could not be mapped."; break;

            case cudaErrorUnmapBufferObjectFailed  : // 15,
                msg = "This indicates that the buffer object could not be unmapped."; break;  

            case cudaErrorInvalidHostPointer       : // 16,
                msg = "This indicates that at least one host pointer passed to the API call is not a valid host pointer."; break;

            case cudaErrorInvalidDevicePointer     : // 17,
                msg = "This indicates that at least one device pointer passed to the API call is not a valid device pointer."; break;
              
            case cudaErrorInvalidTexture           : // 18,
                msg = "This indicates that the texture passed to the API call is not a valid texture."; break;  

            case cudaErrorInvalidTextureBinding    : // 19,
                msg = "This indicates that the texture binding is not valid. This occurs if you call ::cudaGetTextureAlignmentOffset() with an unbound texture."; break;

            case cudaErrorInvalidChannelDescriptor : // 20,
                msg = "This indicates that the channel descriptor passed to the API call is not valid. This occurs if the format is not one of the formats specified by ::cudaChannelFormatKind, or if one of the dimensions is invalid."; break;

            case cudaErrorInvalidMemcpyDirection   : // 21,
                msg = "This indicates that the direction of the memcpy passed to the API call is not one of the types specified by ::cudaMemcpyKind."; break;

            case cudaErrorAddressOfConstant        : // 22,
                msg = "This indicated that the user has taken the address of a constant variable, which was forbidden up until the CUDA 3.1 release. [deprecated] This error return is deprecated as of CUDA 3.1. Variables in constant memory may now have their address taken by the runtime via ::cudaGetSymbolAddress()."; break;

            case cudaErrorTextureFetchFailed       : // 23,
                msg = "This indicated that a texture fetch was not able to be performed. This was previously used for device emulation of texture operations. [deprecated] This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release."; break;

            case cudaErrorTextureNotBound          : // 24,
                msg = "This indicated that a texture was not bound for access. This was previously used for device emulation of texture operations. [deprecated] This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release."; break;

            case cudaErrorSynchronizationError     : // 25,
                msg = "This indicated that a synchronization operation had failed. This was previously used for some device emulation functions. [deprecated] This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release."; break;

            case cudaErrorInvalidFilterSetting     : // 26, 
                msg = "This indicates that a non-float texture was being accessed with linear filtering. This is not supported by CUDA."; break;

            case cudaErrorInvalidNormSetting       : // 27, 
                msg = "This indicates that an attempt was made to read a non-float texture as a normalized float. This is not supported by CUDA."; break;

            case cudaErrorMixedDeviceExecution     : // 28,
                msg = "Mixing of device and device emulation code was not allowed. [deprecated] This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release."; break;

            case cudaErrorCudartUnloading          : // 29,
                msg = "This indicates that a CUDA Runtime API call cannot be executed because it is being called during process shut down, at a point in time after CUDA driver has been unloaded."; break;

            case cudaErrorUnknown                  : // 30,
                msg = "This indicates that an unknown internal error has occurred."; break;

            case cudaErrorNotYetImplemented        : // 31,
                msg = "This indicates that the API call is not yet implemented. Production releases of CUDA will never return this error. [deprecated] This error return is deprecated as of CUDA 4.1."; break;

            case cudaErrorMemoryValueTooLarge      : // 32,
                msg = "his indicated that an emulated device pointer exceeded the 32-bit address range. [deprecated] This error return is deprecated as of CUDA 3.1. Device emulation mode was removed with the CUDA 3.1 release."; break;

            case cudaErrorInvalidResourceHandle    : // 33,
                msg = "This indicates that a resource handle passed to the API call was not valid. Resource handles are opaque types like ::cudaStream_t and ::cudaEvent_t."; break;

            case cudaErrorNotReady                 : // 34,
                msg = "This indicates that asynchronous operations issued previously have not completed yet. This result is not actually an error, but must be indicated differently than ::cudaSuccess (which indicates completion). Calls that may return this value include ::cudaEventQuery() and ::cudaStreamQuery()."; break;

            case cudaErrorInsufficientDriver       : // 35,
                msg = "This indicates that the installed NVIDIA CUDA driver is older than the CUDA runtime library. This is not a supported configuration. Users should install an updated NVIDIA display driver to allow the application to run."; break;

            case cudaErrorSetOnActiveProcess       : // 36,
                msg = "This indicates that the user has called ::cudaSetValidDevices(), ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(), ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by calling non-device management operations (allocating memory and launching kernels are examples of non-device management operations). This error can also be returned if using runtime/driver interoperability and there is an existing ::CUcontext active on the host thread."; break;

            case cudaErrorInvalidSurface           : // 37,
                msg = "This indicates that the surface passed to the API call is not a valid surface."; break;

            case cudaErrorNoDevice                 : // 38,
                msg = "This indicates that no CUDA-capable devices were detected by the installed CUDA driver."; break;

            case cudaErrorECCUncorrectable         : // 39,
                msg = "This indicates that an uncorrectable ECC error was detected during execution."; break;

            case cudaErrorSharedObjectSymbolNotFound   : // 40,
                msg = "This indicates that a link to a shared object failed to resolve."; break;

            case cudaErrorSharedObjectInitFailed   : // 41,
                msg = "This indicates that initialization of a shared object failed."; break;

            case cudaErrorUnsupportedLimit         : // 42,
                msg = "This indicates that the ::cudaLimit passed to the API call is not supported by the active device."; break;

            case cudaErrorDuplicateVariableName    : // 43,
                msg = "This indicates that multiple global or constant variables (across separate CUDA source files in the application) share the same string name."; break;

            case cudaErrorDuplicateTextureName     : // 44,
                msg = "This indicates that multiple textures (across separate CUDA source files in the application) share the same string name."; break;
              
            case cudaErrorDuplicateSurfaceName     : // 45,
                msg = "This indicates that multiple surfaces (across separate CUDA source files in the application) share the same string name."; break;

            case cudaErrorDevicesUnavailable       : // 46,
                msg = "This indicates that all CUDA devices are busy or unavailable at the current time. Devices are often busy/unavailable due to use of ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long running CUDA kernels have filled up the GPU and are blocking new work from starting. They can also be unavailable due to memory constraints on a device that already has active CUDA work being performed."; break;

            case cudaErrorInvalidKernelImage       : // 47,
                msg = "This indicates that the device kernel image is invalid."; break;
              
            case cudaErrorNoKernelImageForDevice   : // 48,
                msg = "there is no kernel image available that is suitable for the device. This can occur when a user specifies code generation options for a particular CUDA source file that do not include the corresponding device configuration."; break;
              
            case cudaErrorIncompatibleDriverContext: // 49,
                msg = "the current context is not compatible with this the CUDA Runtime. This can only occur if you are using CUDA Runtime/Driver interoperability and have created an existing Driver context using the driver API. The Driver context may be incompatible either because the Driver context was created using an older version of the API, because the Runtime API call expects a primary driver context and the Driver context is not primary, or because the Driver context has been destroyed. Please see ref CUDART_DRIVER Interactions with the CUDA Driver API for more information."; break;
                  
            case cudaErrorPeerAccessAlreadyEnabled : // 50,
                msg = "a call to ::cudaDeviceEnablePeerAccess() is trying to re-enable peer addressing on from a context which has already had peer addressing enabled."; break;

            case cudaErrorPeerAccessNotEnabled     : // 51,
                msg = "::cudaDeviceDisablePeerAccess() is trying to disable peer addressing which has not been enabled yet via ::cudaDeviceEnablePeerAccess()."; break;
                
            case cudaErrorDeviceAlreadyInUse       : // 54,
                msg = "a call tried to access an exclusive-thread device that is already in use by a different thread."; break;

            case cudaErrorProfilerDisabled         : // 55,
                msg = "This indicates profiler is not initialized for this run. This can happen when the application is running with external profiling tools like visual profiler."; break;

            case cudaErrorProfilerNotInitialized   : // 56,
                msg = "[deprecated] This error return is deprecated as of CUDA 5.0. It is no longer an error to attempt to enable/disable the profiling via ::cudaProfilerStart or ::cudaProfilerStop without initialization."; break;

            case cudaErrorProfilerAlreadyStarted   : // 57,
                msg = "[deprecated] This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStart() when profiling is already enabled."; break;

            case cudaErrorProfilerAlreadyStopped   : //58,
                msg = "[deprecated] This error return is deprecated as of CUDA 5.0. It is no longer an error to call cudaProfilerStop() when profiling is already disabled."; break;

            case cudaErrorAssert                    : //59,
                msg = "An assert triggered in device code during kernel execution. The device cannot be used again until ::cudaThreadExit() is called. All existing allocations are invalid and must be reconstructed if the program is to continue using CUDA."; break;
              
            case cudaErrorTooManyPeers             : // 60,
                msg = "the hardware resources required to enable peer access have been exhausted for one or more of the devices passed to ::cudaEnablePeerAccess()."; break;
              
            case cudaErrorHostMemoryAlreadyRegistered: // 61,
                msg = "the memory range passed to ::cudaHostRegister() has already been registered."; break;
                    
            case cudaErrorHostMemoryNotRegistered  : // 62,
                msg = "the pointer passed to ::cudaHostUnregister() does not correspond to any currently registered memory region."; break;

            case cudaErrorOperatingSystem          : // 63,
                msg = "an OS call failed."; break;

            case cudaErrorPeerAccessUnsupported    : // 64,
                msg = "P2P access is not supported across the given devices."; break;

            case cudaErrorLaunchMaxDepthExceeded   : // 65,
                msg = "a device runtime grid launch did not occur because the depth of the child grid would exceed the maximum supported number of nested grid launches."; break;

            case cudaErrorLaunchFileScopedTex      : // 66,
                msg = "a grid launch did not occur because the kernel uses file-scoped textures which are unsupported by the device runtime. Kernels launched via the device runtime only support textures created with the Texture Object API's."; break;

            case cudaErrorLaunchFileScopedSurf     : // 67,
                msg = "a grid launch did not occur because the kernel uses file-scoped surfaces which are unsupported by the device runtime. Kernels launched via the device runtime only support surfaces created with the Surface Object API's."; break;

            case cudaErrorSyncDepthExceeded        : // 68,
                msg = "a call to ::cudaDeviceSynchronize made from the device runtime failed because the call was made at grid depth greater than than either the default (2 levels of grids) or user specified device limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on launched grids at a greater depth successfully, the maximum nested depth at which ::cudaDeviceSynchronize will be called must be specified with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit api before the host-side launch of a kernel using the device runtime. Keep in mind that additional levels of sync depth require the runtime to reserve large amounts of device memory that cannot be used for user allocations."; break;

            case cudaErrorLaunchPendingCountExceeded : //     69,
                msg = "a device runtime grid launch failed because the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount. For this launch to proceed successfully, ::cudaDeviceSetLimit must be called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher than the upper bound of outstanding launches that can be issued to the device runtime. Keep in mind that raising the limit of pending device runtime launches will require the runtime to reserve device memory that cannot be used for user allocations."; break;    

            case cudaErrorNotPermitted             : // 70,
                msg = "This error indicates the attempted operation is not permitted."; break;

            case cudaErrorNotSupported             : // 71,
                msg = "This error indicates the attempted operation is not supported on the current system or device."; break;

            case cudaErrorHardwareStackError       : // 72,
                msg = "Device encountered an error in the call stack during kernel execution, possibly due to stack corruption or exceeding the stack size limit. The context cannot be used, so it must be destroyed (and a new one should be created). All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA."; break;

            case cudaErrorIllegalInstruction       : // 73,
                msg = "The device encountered an illegal instruction during kernel execution The context cannot be used, so it must be destroyed (and a new one should be created). All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA."; break;

            case cudaErrorMisalignedAddress        : // 74,
                msg = "The device encountered a load or store instruction on a memory address which is not aligned. The context cannot be used, so it must be destroyed (and a new one should be created). All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA."; break;

            case cudaErrorInvalidAddressSpace      : // 75,
                msg = "While executing a kernel, the device encountered an instruction which can only operate on memory locations in certain address spaces (global, shared, or local), but was supplied a memory address not belonging to an allowed address space. The context cannot be used, so it must be destroyed (and a new one should be created). All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA."; break;

            case cudaErrorInvalidPc                : // 76,
                msg = "The device encountered an invalid program counter. The context cannot be used, so it must be destroyed (and a new one should be created). All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA."; break;

            case cudaErrorIllegalAddress           : // 77,
                msg = "The device encountered a load or store instruction on an invalid memory address. The context cannot be used, so it must be destroyed (and a new one should be created). All existing device memory allocations from this context are invalid and must be reconstructed if the program is to continue using CUDA."; break;

            case cudaErrorInvalidPtx               : // 78,
                msg = "A PTX compilation failed. The runtime may fall back to compiling PTX if an application does not contain a suitable binary for the current device."; break;

            case cudaErrorInvalidGraphicsContext   : // 79,
                msg = "This indicates an error with the OpenGL or DirectX context."; break;

            case cudaErrorStartupFailure   : //  0x7f,
                msg = "This indicates an internal startup failure in the CUDA runtime."; break;

            case cudaErrorApiFailureBase   : //  10000
                msg = "Any unhandled CUDA driver error is added to this value and returned via the runtime. Production releases of CUDA should not return such errors. [deprecated] This error return is deprecated as of CUDA 4.1."; break;

        }

        std::cerr << msg << std::endl;

        FatalError();
    }
}

void checkCUDNN(const int lineNumber, cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "CUDNN failure at LINE " << lineNumber << ": ";
        switch (status) {
            case CUDNN_STATUS_SUCCESS:              std::cerr << "CUDNN_STATUS_SUCCESS" << std::endl; break;
            case CUDNN_STATUS_NOT_INITIALIZED:      std::cerr << "CUDNN_STATUS_NOT_INITIALIZED" << std::endl; break;
            case CUDNN_STATUS_ALLOC_FAILED:         std::cerr << "CUDNN_STATUS_ALLOC_FAILED" << std::endl; break;
            case CUDNN_STATUS_BAD_PARAM:            std::cerr << "CUDNN_STATUS_BAD_PARAM" << std::endl; break;
            case CUDNN_STATUS_INTERNAL_ERROR:       std::cerr << "CUDNN_STATUS_INTERNAL_ERROR" << std::endl; break;
            case CUDNN_STATUS_INVALID_VALUE:        std::cerr << "CUDNN_STATUS_INVALID_VALUE" << std::endl; break;
            case CUDNN_STATUS_ARCH_MISMATCH:        std::cerr << "CUDNN_STATUS_ARCH_MISMATCH" << std::endl; break;
            case CUDNN_STATUS_MAPPING_ERROR:        std::cerr << "CUDNN_STATUS_MAPPING_ERROR" << std::endl; break;
            case CUDNN_STATUS_EXECUTION_FAILED:     std::cerr << "CUDNN_STATUS_EXECUTION_FAILED" << std::endl; break;
            case CUDNN_STATUS_NOT_SUPPORTED:        std::cerr << "CUDNN_STATUS_NOT_SUPPORTED" << std::endl; break;
            case CUDNN_STATUS_LICENSE_ERROR:        std::cerr << "CUDNN_STATUS_LICENSE_ERROR" << std::endl; break;
        }
        FatalError();
    }
    checkCUDA(lineNumber,cudaGetLastError());

}
void checkCUBLAS(const int lineNumber, cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS failure at LINE " << lineNumber << ": ";
        switch (status) {
            case CUBLAS_STATUS_SUCCESS:             std::cerr << "CUBLAS_STATUS_SUCCESS" << std::endl; break;
            case CUBLAS_STATUS_NOT_INITIALIZED:     std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED" << std::endl; break;
            case CUBLAS_STATUS_ALLOC_FAILED:        std::cerr << "CUBLAS_STATUS_ALLOC_FAILED" << std::endl; break;
            case CUBLAS_STATUS_INVALID_VALUE:       std::cerr << "CUBLAS_STATUS_INVALID_VALUE" << std::endl; break;
            case CUBLAS_STATUS_ARCH_MISMATCH:       std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH" << std::endl; break;
            case CUBLAS_STATUS_MAPPING_ERROR:       std::cerr << "CUBLAS_STATUS_MAPPING_ERROR" << std::endl; break;
            case CUBLAS_STATUS_EXECUTION_FAILED:    std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED" << std::endl; break;
            case CUBLAS_STATUS_INTERNAL_ERROR:      std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR" << std::endl; break;
            case CUBLAS_STATUS_NOT_SUPPORTED:       std::cerr << "CUBLAS_STATUS_NOT_SUPPORTED" << std::endl; break;
            case CUBLAS_STATUS_LICENSE_ERROR:       std::cerr << "CUBLAS_STATUS_LICENSE_ERROR" << std::endl; break;
        }
        FatalError();
    }
    checkCUDA(lineNumber,cudaGetLastError());
}

unsigned long long get_timestamp() {
    struct timeval now;
    gettimeofday (&now, NULL);
    return  now.tv_usec + (unsigned long long)now.tv_sec * 1000000;
}

unsigned long long ticBegin;

unsigned long long tic() {
    ticBegin = get_timestamp();
    return ticBegin;
}

unsigned long long toc() {
    unsigned long long ticEnd = get_timestamp();
    unsigned long long delta = ticEnd - ticBegin;
    std::cout << "Time passes " << delta << " microseconds" <<std::endl;
    ticBegin = ticEnd;
    return delta;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// HALF computation ultility
//////////////////////////////////////////////////////////////////////////////////////////////////

static __inline__ __device__ __host__ int ishnan(half h) {
    // When input is NaN, exponent is all ones and mantissa is non-zero.
    return (h.x & 0x7c00U) == 0x7c00U && (h.x & 0x03ffU) != 0;
}

half cpu_float2half(float f) {
    half ret;

    unsigned x = *((int*)(void*)(&f));
    unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
    unsigned sign, exponent, mantissa;

    // Get rid of +NaN/-NaN case first.
    if (u > 0x7f800000) {
        ret.x = 0x7fffU;
        return ret;
    }

    sign = ((x >> 16) & 0x8000);

    // Get rid of +Inf/-Inf, +0/-0.
    if (u > 0x477fefff) {
        ret.x = sign | 0x7c00U;
        return ret;
    }
    if (u < 0x33000001) {
        ret.x = (sign | 0x0000);
        return ret;
    }

    exponent = ((u >> 23) & 0xff);
    mantissa = (u & 0x7fffff);

    if (exponent > 0x70) {
        shift = 13;
        exponent -= 0x70;
    } else {
        shift = 0x7e - exponent;
        exponent = 0;
        mantissa |= 0x800000;
    }
    lsb = (1 << shift);
    lsb_s1 = (lsb >> 1);
    lsb_m1 = (lsb - 1);

    // Round to nearest even.
    remainder = (mantissa & lsb_m1);
    mantissa >>= shift;
    if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
        ++mantissa;
        if (!(mantissa & 0x3ff)) {
            ++exponent;
            mantissa = 0;
        }
    }

    ret.x = (sign | (exponent << 10) | mantissa);

    return ret;
}


float cpu_half2float(half h) {
    unsigned sign = ((h.x >> 15) & 1);
    unsigned exponent = ((h.x >> 10) & 0x1f);
    unsigned mantissa = ((h.x & 0x3ff) << 13);

    if (exponent == 0x1f) {  /* NaN or Inf */
        mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
        exponent = 0xff;
    } else if (!exponent) {  /* Denorm or Zero */
        if (mantissa) {
            unsigned int msb;
            exponent = 0x71;
            do {
                msb = (mantissa & 0x400000);
                mantissa <<= 1;  /* normalize */
                --exponent;
            } while (!msb);
            mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70;
    }

    int temp = ((sign << 31) | (exponent << 23) | mantissa);

    return *((float*)((void*)&temp));
}


bool operator <(const half& x, const half& y) {
    return cpu_half2float(x) < cpu_half2float(y);
}

std::ostream& operator<< (std::ostream& stream, const half& x) {
    stream << cpu_half2float(x);
    return stream;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// JSON parser
//////////////////////////////////////////////////////////////////////////////////////////////////

enum JSONType { JSON_String, JSON_Bool, JSON_Null, JSON_Number, JSON_Object, JSON_ObjectArray};

// plain object
class JSON{
public:
    JSONType type;
    std::vector<void*> array;
    std::map<std::string, JSON*> member;

    ~JSON(){
        for (int i=0;i<array.size();++i){
            if (array[i]!=NULL){
                switch(type){
                    case JSON_String:
                        delete ((std::string*)(array[i]));
                    break;
                    case JSON_Bool:
                        delete ((bool*)(array[i]));
                    break;
                    case JSON_Null:
                    break;
                    case JSON_Number:
                        delete ((ComputeT*)(array[i]));
                    break;
                    case JSON_Object:
                    break;
                    case JSON_ObjectArray:
                        delete ((JSON*)(array[i]));
                    break;
                }
            }
        }
        for (std::map<std::string, JSON*>::iterator it = member.begin(); it != member.end(); it++ ){
            if (it->second != NULL)
                delete it->second;
        }
    };

    std::string returnString(){
        if (type!=JSON_String) FatalError(__LINE__);
        return *((std::string*)(array[0]));
    };

    bool returnBool(){
        if (type!=JSON_Bool) FatalError(__LINE__);
        return *((bool*)(array[0]));
    };

    ComputeT returnReal(){
        if (type!=JSON_Number) FatalError(__LINE__);
        return *((ComputeT*)(array[0]));
    };

    std::vector<int> returnIntVector(){
        if (type!=JSON_Number) FatalError(__LINE__);
        std::vector<int> v(array.size());
        for (int i=0;i<array.size();++i){
            v[i] = (int)(*((ComputeT*)(array[i])));
        }
        return v;
    };

    std::vector<ComputeT> returnRealVector(){
        if (type!=JSON_Number) FatalError(__LINE__);
        std::vector<ComputeT> v(array.size());
        for (int i=0;i<array.size();++i){
            v[i] = (ComputeT)(*((ComputeT*)(array[i])));
        }
        return v;
    };

    std::vector<std::string> returnStringVector(){
        if (type!=JSON_String) FatalError(__LINE__);
        std::vector<std::string> v(array.size());
        for (int i=0;i<array.size();++i){
            v[i] = *((std::string*)(array[i]));
        }
        return v;
    };

    void setOrDie(std::string name, unsigned int &variable){
        if (this->member.find(name) == this->member.end()){
            FatalError(__LINE__);
        }
        else variable = (unsigned int)this->member[name]->returnReal();
    };

    void setOrDie(std::string name, bool &variable){
        if (this->member.find(name) == this->member.end()){
            FatalError(__LINE__);
        }
        else variable = this->member[name]->returnBool();
    };

    void setOrDie(std::string name, std::vector<ComputeT> &variable){
        if (this->member.find(name) == this->member.end())
            FatalError(__LINE__);
        else variable = this->member[name]->returnRealVector();
    };

    void set(std::string name, bool &variable, bool default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnBool();
    };

    void set(std::string name, ComputeT &variable, ComputeT default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (ComputeT)(this->member[name]->returnReal());
    };

    void setOrDie(std::string name, ComputeT &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = (ComputeT)(this->member[name]->returnReal());
    };

    void set(std::string name, int &variable, int default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (int)(this->member[name]->returnReal());
    };

    void set(std::string name, double &variable, double default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (double)(this->member[name]->returnReal());
    };

    void set(std::string name, unsigned int &variable, unsigned int default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = (unsigned int)(this->member[name]->returnReal());
    };

    void setOrDie(std::string name, int &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = (int)(this->member[name]->returnReal());
    };

    void set(std::string name, std::vector<int> &variable, std::vector<int> default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnIntVector();
    };

    void set(std::string name, std::vector<ComputeT> &variable, std::vector<ComputeT> default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnRealVector();
    };

    void set(std::string name, std::vector<std::string> &variable, std::vector<std::string> default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnStringVector();
    };

    void setOrDie(std::string name, std::vector<std::string> &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = this->member[name]->returnStringVector();
    };

    void setOrDie(std::string name, std::vector<int> &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = this->member[name]->returnIntVector();
    };

    void set(std::string name, std::string &variable, std::string default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else variable = this->member[name]->returnString();
    };

    void setOrDie(std::string name, std::string &variable){
        if (this->member.find(name) == this->member.end())                              FatalError(__LINE__);
        else variable = this->member[name]->returnString();
    };

    void setOrDie(std::string name, ElementWiseOp &variable){
        if (this->member.find(name) == this->member.end())                                  FatalError(__LINE__);
        else if (0 == this->member[name]->returnString().compare("ElementWise_EQL"))        variable = ElementWise_EQL;
        else if (0 == this->member[name]->returnString().compare("ElementWise_MUL"))        variable = ElementWise_MUL;
        else if (0 == this->member[name]->returnString().compare("ElementWise_SUM"))        variable = ElementWise_SUM;
        else if (0 == this->member[name]->returnString().compare("ElementWise_MIN"))        variable = ElementWise_MIN;
        else if (0 == this->member[name]->returnString().compare("ElementWise_MAX"))        variable = ElementWise_MAX;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, Filler &variable, Filler default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Xavier"))             variable = Xavier;
        else if (0 == this->member[name]->returnString().compare("Gaussian"))           variable = Gaussian;
        else if (0 == this->member[name]->returnString().compare("Constant"))           variable = Constant;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, Pool &variable, Pool default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Max"))                variable = Max;
        else if (0 == this->member[name]->returnString().compare("Average"))            variable = Average;
        else if (0 == this->member[name]->returnString().compare("Sum"))                variable = Sum;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void setOrDie(std::string name, LossObjective &variable){
        if (this->member.find(name) == this->member.end())                                                  FatalError(__LINE__);
        else if (0 == this->member[name]->returnString().compare("MultinomialLogistic_StableSoftmax"))      variable = MultinomialLogistic_StableSoftmax;
        else if (0 == this->member[name]->returnString().compare("MultinomialLogistic"))                    variable = MultinomialLogistic;
        else if (0 == this->member[name]->returnString().compare("SmoothL1"))                               variable = SmoothL1;
        else if (0 == this->member[name]->returnString().compare("Contrastive"))                            variable = Contrastive;
        else if (0 == this->member[name]->returnString().compare("EuclideanSSE"))                           variable = EuclideanSSE;
        else if (0 == this->member[name]->returnString().compare("HingeL1"))                                variable = HingeL1;
        else if (0 == this->member[name]->returnString().compare("HingeL2"))                                variable = HingeL2;
        else if (0 == this->member[name]->returnString().compare("SigmoidCrossEntropy"))                    variable = SigmoidCrossEntropy;
        else if (0 == this->member[name]->returnString().compare("Infogain"))                               variable = Infogain;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, Phase &variable, Phase default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Training"))           variable = Training;
        else if (0 == this->member[name]->returnString().compare("Testing"))            variable = Testing;
        else if (0 == this->member[name]->returnString().compare("TrainingTesting"))    variable = TrainingTesting;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, LRPolicy &variable, LRPolicy default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("LR_fixed"))           variable = LR_fixed;
        else if (0 == this->member[name]->returnString().compare("LR_step"))            variable = LR_step;
        else if (0 == this->member[name]->returnString().compare("LR_exp"))             variable = LR_exp;
        else if (0 == this->member[name]->returnString().compare("LR_inv"))             variable = LR_inv;
        else if (0 == this->member[name]->returnString().compare("LR_multistep"))       variable = LR_multistep;
        else if (0 == this->member[name]->returnString().compare("LR_poly"))            variable = LR_poly;
        else if (0 == this->member[name]->returnString().compare("LR_sigmoid"))         variable = LR_sigmoid;
        else if (0 == this->member[name]->returnString().compare("LR_cyclical"))        variable = LR_cyclical;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, SolverAlgorithm &variable, SolverAlgorithm default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("SGD"))                variable = SGD;
        else if (0 == this->member[name]->returnString().compare("AdaDelta"))           variable = AdaDelta;
        else if (0 == this->member[name]->returnString().compare("AdaGrad"))            variable = AdaGrad;
        else if (0 == this->member[name]->returnString().compare("Adam"))               variable = Adam;
        else if (0 == this->member[name]->returnString().compare("NAG"))                variable = NAG;
        else if (0 == this->member[name]->returnString().compare("RMSprop"))            variable = RMSprop;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, Regularizer &variable, Regularizer default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("L2"))                 variable = L2;
        else if (0 == this->member[name]->returnString().compare("L1"))                 variable = L1;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, LRN &variable, LRN default_value){
        if (this->member.find(name) == this->member.end())                                  variable = default_value;
        else if (0 == this->member[name]->returnString().compare("CrossChannel"))           variable = CrossChannel;
        else if (0 == this->member[name]->returnString().compare("DivisiveNormalization"))  variable = DivisiveNormalization;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, cudnnBatchNormMode_t &variable, cudnnBatchNormMode_t default_value){
        if (this->member.find(name) == this->member.end())                                  variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Spatial"))                variable = CUDNN_BATCHNORM_SPATIAL;
        else if (0 == this->member[name]->returnString().compare("PerActivation"))          variable = CUDNN_BATCHNORM_PER_ACTIVATION;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, cudnnPoolingMode_t &variable, cudnnPoolingMode_t default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("max"))                variable = CUDNN_POOLING_MAX;
        else if (0 == this->member[name]->returnString().compare("average_include"))    variable = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        else if (0 == this->member[name]->returnString().compare("average_exclude"))    variable = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, cudnnActivationMode_t &variable, cudnnActivationMode_t default_value){
        if (this->member.find(name) == this->member.end())                              variable = default_value;
        else if (0 == this->member[name]->returnString().compare("Sigmoid"))            variable = CUDNN_ACTIVATION_SIGMOID;
        else if (0 == this->member[name]->returnString().compare("ReLU"))               variable = CUDNN_ACTIVATION_RELU;
        else if (0 == this->member[name]->returnString().compare("TanH"))               variable = CUDNN_ACTIVATION_TANH;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, cudnnConvolutionFwdAlgo_t &variable, cudnnConvolutionFwdAlgo_t default_value){
        if (this->member.find(name) == this->member.end())                                  variable = default_value;
        else if (0 == this->member[name]->returnString().compare("implicit_gemm"))          variable = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        else if (0 == this->member[name]->returnString().compare("implicit_precomp_gemm"))  variable = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        else if (0 == this->member[name]->returnString().compare("gemm"))                   variable = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
        else if (0 == this->member[name]->returnString().compare("direct"))                 variable = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
        else if (0 == this->member[name]->returnString().compare("fft"))                    variable = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
        else if (0 == this->member[name]->returnString().compare("fft_tiling"))             variable = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
        else if (0 == this->member[name]->returnString().compare("winograd"))               variable = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        else if (0 == this->member[name]->returnString().compare("winograd_nonfused"))      variable = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, cudnnConvolutionBwdDataAlgo_t &variable, cudnnConvolutionBwdDataAlgo_t default_value){
        if (this->member.find(name) == this->member.end())                                  variable = default_value;
        else if (0 == this->member[name]->returnString().compare("0"))                      variable = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        else if (0 == this->member[name]->returnString().compare("1"))                      variable = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
        else if (0 == this->member[name]->returnString().compare("fft"))                    variable = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
        else if (0 == this->member[name]->returnString().compare("fft_tiling"))             variable = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
        else if (0 == this->member[name]->returnString().compare("winograd"))               variable = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
        else if (0 == this->member[name]->returnString().compare("winograd_nonfused"))      variable = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void set(std::string name, cudnnConvolutionBwdFilterAlgo_t &variable, cudnnConvolutionBwdFilterAlgo_t default_value){
        if (this->member.find(name) == this->member.end())                                  variable = default_value;
        else if (0 == this->member[name]->returnString().compare("0"))                      variable = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        else if (0 == this->member[name]->returnString().compare("1"))                      variable = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
        else if (0 == this->member[name]->returnString().compare("fft"))                    variable = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
        else if (0 == this->member[name]->returnString().compare("3"))                      variable = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
        else if (0 == this->member[name]->returnString().compare("winograd_nonfused"))      variable = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
        else{ std::cout<<"Unsupported "<<name<<" = "<<this->member[name]->returnString()<<std::endl; FatalError(__LINE__); }
    };

    void print(){
        switch(type){
            case JSON_String:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << "\"" << *((std::string*)(array[i])) << "\""  ;
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Bool:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << ((*((bool*)(array[i])))? "true": "false");
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Null:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << "null";
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Number:
                if (array.size()>1) std::cout<<"[";
                for (int i=0;i<array.size();++i){
                    if (i>0) std::cout<< ",";
                    std::cout << *((ComputeT*)(array[i]));
                }
                if (array.size()>1) std::cout<<"]";
                std::cout<<std::endl;
            break;
            case JSON_Object:
                std::cout<<"{"<<std::endl;
                for (std::map<std::string, JSON*>::iterator it = member.begin(); it != member.end(); it++ ){
                    std::cout << "\t" << it->first << ": ";
                    it->second->print();
                }
                std::cout<<"}";
            break;
            case JSON_ObjectArray:
                std::cout<<"["<<std::endl;
                for (int i=0;i<array.size();++i){
                    JSON* p = (JSON*)(array[i]);
                    p->print();
                    if (i<array.size()-1) std::cout<<","<<std::endl;
                }
                std::cout<<"]"<<std::endl;
            break;
        }
    };

    void parseNumberOrTextArray(std::string input){
        while (input.size()>0){
            int e = input.find(",");
            if (e==std::string::npos){
                e = input.size();
            }
            std::string first = input.substr(0,e);
            if (first[0]=='\"'){
                type = JSON_String;
                std::string* p = new std::string(first.substr(1,first.size()-2));
                array.push_back((void*)p);
            }else if (first[0]=='t'){
                type = JSON_Bool;
                bool* p = new bool(true);
                array.push_back((void*)p);
            }else if (first[0]=='f'){
                type = JSON_Bool;
                bool* p = new bool(false);
                array.push_back((void*)p);
            }else if (first[0]=='n'){
                type = JSON_Null;
                void* p = NULL;
                array.push_back((void*)p);
            }else{
                type = JSON_Number;
                ComputeT* p = new ComputeT(stof(first));
                array.push_back((void*)p);
            }
            if(e+1<input.size())
                input=input.substr(e+1);
            else
                break;
        }
    };

    void parseObject(std::string input){
        type = JSON_Object;
        int b,m,e;
        JSON* p;
        b = input.find("{");
        e = input.find("}");
        input = input.substr(b+1,e-b-1);

        while (true){
            m= input.find(":");
            if (std::string::npos==m) break;

            std::string name = input.substr(0,m);
            name = name.substr(1,m-2);
            input = input.substr(m+1);
            if (input[0]=='\"'){
                e=input.find("\"",1);
                p = new JSON;
                p->parseNumberOrTextArray(input.substr(0,e+1));
                this->member[name] = p;

                if (e+2<input.size())
                    input = input.substr(e+2);
                else
                    break;
            }else if (input[0]=='['){
                // assume no nested array
                input = input.substr(1);
                e = input.find("]");
                p = new JSON;
                p->parseNumberOrTextArray(input.substr(0,e));
                this->member[name] = p;

                if (e+1<input.size())
                    input = input.substr(e+2);
                else
                    break;
            }else if (input[0]=='f' || input[0]=='t' || input[0]=='.' || input[0]=='-' || ('0'<=input[0] && input[0]<='9')){
                e=input.find(",");
                if (e==std::string::npos){
                    e = input.size();
                }
                p = new JSON;
                p->parseNumberOrTextArray(input.substr(0,e));
                this->member[name] = p;

                if (e+1<input.size())
                    input = input.substr(e+1);
                else
                    break;
            }else{
                FatalError(__LINE__);
            }
        }
    };
    void parseObjectArray(std::string input){
        type = JSON_ObjectArray;

        input = input.substr(1,input.size()-2);

        while (input.size()>0){
            int e = input.find("}")+1;
            if (e==std::string::npos){
                e = input.size();
            }
            std::string first = input.substr(0,e);
            JSON* pObj = new JSON;
            pObj->parseObject(first);
            array.push_back((void*)pObj);

            if(e+1<input.size())
                input=input.substr(e+1);
            else
                break;
        }
    };
};

#define SetValue(obj,attribute,value) obj->set(#attribute,attribute,value);
#define SetOrDie(obj,attribute)       obj->setOrDie(#attribute,attribute);


void parseNetworkJSON(std::string filename, JSON* train_obj, JSON* test_obj, JSON* architecture_obj){
    std::ifstream t(filename);
    std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    str.erase(remove_if(str.begin(), str.end(), (int(*)(int))isspace), str.end());
    std::string input = str;
    int b,e;

    b = input.find("\"train\"");
    std::string train_str = input.substr(b+7);
    b = train_str.find("{");
    e = train_str.find("}");
    train_str=train_str.substr(b,e-b+1);
    if (train_obj!=NULL) train_obj->parseObject(train_str);

    b = input.find("\"test\"");
    std::string test_str = input.substr(b+6);
    b = test_str.find("{");
    e = test_str.find("}");
    test_str=test_str.substr(b,e-b+1);
    if (test_obj!=NULL) test_obj->parseObject(test_str);

    b=input.find("\"layers\"");
    input = input.substr(b+9);
    e=input.find("}]");
    if (architecture_obj!=NULL) architecture_obj->parseObjectArray(input);

}


//////////////////////////////////////////////////////////////////////////////////////////////////
// Utility
//////////////////////////////////////////////////////////////////////////////////////////////////

bool is_file_exist(const std::string& fileName){
    std::ifstream infile(fileName);
    return infile.good();
}

void memorySizePrint(size_t bytes){
    if (bytes<512){
        std::cout<<bytes<<" Bytes";
    }else if (bytes<512.0*1024){
        std::cout<<(bytes/1024.0)<<" KB";
    }else if (bytes<512.0*1024*1024){
        std::cout<<(bytes/(1024.0*1024.0))<<" MB";
    }else if (bytes<512.0*1024*1024*1024){
        std::cout<<(bytes/(1024.0*1024.0*1024.0))<<" GB";
    }else if (bytes<512.0*1024*1024*1024*1024){
        std::cout<<(bytes/(1024.0*1024.0*1024.0*1024.0))<<" TB";
    }else{
        std::cout<<(bytes/(1024.0*1024.0*1024.0*1024.0*1024.0))<<" PB";
    }
}

void veciPrint(const std::vector<int>& v){
    std::cout<<"["<<v.size()<<"]={";
    if (v.size()>0) std::cout<<v[0];
    if (v.size()>1){
        for (int i=1;i<v.size();++i){
            std::cout<<","<<v[i];
        }
    }
    std::cout<<"}";
}

void vecfPrint(const std::vector<ComputeT>& v){
    std::cout<<"[";
    if (v.size()>0) std::cout<<v[0];
    if (v.size()>1){
        for (int i=1;i<v.size();++i){
            std::cout<<","<<v[i];
        }
    }
    std::cout<<"]";
}

std::vector<int> veci(int n, ...){
    std::vector<int> v;
    if (n==0) return v;
    va_list ap;
    va_start(ap, n);
    for(int i = 0; i < n; i++) {
        v.push_back(va_arg(ap, int));
    }
    va_end(ap);
    return v;
}

std::vector<std::string> vecs(int n, ...){
    std::vector<std::string> v;
    if (n==0) return v;
    va_list ap;
    va_start(ap, n);
    for(int i = 0; i < n; i++) {
        v.push_back(std::string(va_arg(ap, char*)));
    }
    va_end(ap);
    return v;
}

std::vector<std::string> getStringVector(std::string input){
    std::vector<std::string> ret;
    while (input.size()>0){
        int e = input.find(",");
        if (e==std::string::npos){
            e = input.size();
        }
        std::string first = input.substr(0,e);
        ret.push_back(first);
        if(e+1<input.size())
            input=input.substr(e+1);
        else
            break;
    }
    return ret;
}

std::vector<std::vector<int> > getIntVectorVector(std::string input){
    //remove all space
    input.erase(remove_if(input.begin(), input.end(), (int(*)(int))isspace), input.end());

    std::vector<std::vector<int> > ret;
    while (input.size()>0){
        int e;
        if (input[0]=='['){
            ret.resize(ret.size()+1);
            e=0;
        }else if (input[0]==','){
            e=0;
        }else if (input[0]==']'){
            e=0;
        }else{
            e = input.find(",");
            if (e==std::string::npos){
                e = input.size();
            }
            int f = input.find("]");
            if (f==std::string::npos){
                f = input.size();
            }
            e = min(e,f);
            std::string first = input.substr(0,e);
            ret[ret.size()-1].push_back(stoi(first));
        }
        if(e+1<input.size())
            input=input.substr(e+1);
        else
            break;
    }
    return ret;
}

size_t numel(const std::vector<int>& dim){
    size_t res = 1;
    for (int i=0;i<dim.size();++i) res *= (size_t)(dim[i]);
    return res;
}

size_t sizeofitem(const std::vector<int>& dim){
    size_t res = 1;
    for (int i=1;i<dim.size();++i) res *= (size_t)(dim[i]);
    return res;
}

size_t numspel(const std::vector<int>& dim){
    size_t res = 1;
    for (int i=2;i<dim.size();++i) res *= (size_t)(dim[i]);
    return res;
}

bool same_dim(const std::vector<int>& dimA, const std::vector<int>& dimB){
    if (dimA.size()!=dimB.size()) return false;
    for (int i=0;i<dimA.size();++i){
        if (dimA[i]!=dimB[i]) return false;
    }
    return true;
}

bool same_dim_EC(const std::vector<int>& dimA, const std::vector<int>& dimB){
    if (dimA.size()!=dimB.size()) return false;
    if (dimA[0]!=dimB[0]) return false;
    for (int i=2;i<dimA.size();++i)
        if (dimA[i]!=dimB[i])
            return false;
    return true;
}

size_t checkNaN(StorageT* dataGPU, size_t n){
    StorageT* CPUmem = new StorageT[n];
    cudaMemcpy(CPUmem, dataGPU, n*sizeofStorageT, cudaMemcpyDeviceToHost);
    size_t countNaN = 0;
    for (size_t i=0;i<n;++i) if (ISNAN(CPUmem[i])) ++countNaN;
    if (countNaN>0){
        std::cout<<"        checkNaN result: "<<countNaN<<" out of "<<n<<" ("<< 100*ComputeT(countNaN)/n<< "\045) values are NaN, "<<n-countNaN<<" are not NaN."; //<<std::endl;
    }
    delete [] CPUmem;
    return countNaN;
}

std::vector<size_t> randperm(size_t n, std::mt19937& rng){
    std::vector<size_t> v(n);
    for (size_t i=0;i<n;++i) v[i]=i;

    shuffle ( v.begin(), v.end(), rng );
    return v;
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

std::string int_to_str(const int i) {
    std::ostringstream s;
    s << i;
    return s.str();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
//////////////////////////////////////////////////////////////////////////////////////////////////


#define CUDA_NUM_THREADS 512

#define MAX_NUM_BLOCKS 2880

inline int CUDA_GET_BLOCKS(const size_t N) {
    return min(MAX_NUM_BLOCKS, int((N + size_t(CUDA_NUM_THREADS) - 1) / CUDA_NUM_THREADS));
}

inline size_t CUDA_GET_LOOPS(const size_t N) {
    size_t total_threads = CUDA_GET_BLOCKS(N)*CUDA_NUM_THREADS;
    return (N + total_threads -1)/ total_threads;
}

__global__ void Accuracy_MultinomialLogistic(
    size_t CUDA_NUM_LOOPS, size_t N, int C, int M, size_t wN,
    const StorageT *pred, const StorageT *label, const StorageT *weight,
    const StorageT *weightTensor, StorageT *loss) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {
        int l = int(GPUStorage2ComputeT(label[idx]));
        int baseID = (idx / M) * C * M + idx % M;
        int elementID = baseID + l * M;
        ComputeT prob = GPUStorage2ComputeT(pred[elementID]);
        loss[idx] = GPUCompute2StorageT(1);
        for (int d = 0; d < C; ++d) {
            if (GPUStorage2ComputeT(pred[baseID + d * M]) > prob) {
                loss[idx] = GPUCompute2StorageT(0);
            }
        }
        if (weight != NULL) {
            loss[idx] = GPUCompute2StorageT(GPUStorage2ComputeT(loss[idx]) *
                                            GPUStorage2ComputeT(weight[l]));
        }
        if (weightTensor != NULL) {
            loss[idx] = GPUCompute2StorageT(GPUStorage2ComputeT(loss[idx]) *
                                            GPUStorage2ComputeT(
                                                weightTensor[idx % wN]));
        }
    }
}

__global__ void Loss_MultinomialLogistic(
    size_t CUDA_NUM_LOOPS, size_t N, int C, int M, size_t wN,
    const StorageT* pred, const StorageT* label, const StorageT* weight,
    const StorageT *weightTensor, StorageT *loss) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {
        int l = int(GPUStorage2ComputeT(label[idx]));
        int offset = l * M + (idx % M);
        int elementID = (idx / M) * C * M + offset;
        ComputeT prob = max(GPUStorage2ComputeT(pred[elementID]), ComputeT_MIN);
        ComputeT res = log(prob);
        if (weight != NULL) res *= GPUStorage2ComputeT(weight[l]);
        if (weightTensor != NULL)
            res *= GPUStorage2ComputeT(weightTensor[elementID % wN]);
        loss[idx] = GPUCompute2StorageT(res);
    }
}

__global__ void LossGrad_MultinomialLogistic(
    size_t CUDA_NUM_LOOPS, size_t N, int C, int M, size_t wN, ComputeT scale,
    const StorageT *pred, const StorageT *label, const StorageT *weight,
    const StorageT *weightTensor, StorageT *diff) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {
        int l = int(GPUStorage2ComputeT(label[idx]));
        int offset = l * M + (idx % M);
        int elementID = (idx / M) * C * M + offset;
        ComputeT prob = max(GPUStorage2ComputeT(pred[elementID]), ComputeT_MIN);
        if (weight != NULL) scale *= GPUStorage2ComputeT(weight[l]);
        if (weightTensor != NULL)
            scale *= GPUStorage2ComputeT(weightTensor[elementID % wN]);
        diff[elementID] = GPUCompute2StorageT(
            GPUStorage2ComputeT(diff[elementID]) + scale / prob);
    }
}

// for numerical stability: http://freemind.pluskid.org/machine-learning/softmax-vs-softmax-loss-numerical-stability/
__global__ void LossGrad_MultinomialLogistic_StableSoftmax(
    size_t CUDA_NUM_LOOPS, size_t N, int C, int M, size_t wN, ComputeT scale,
    const StorageT *pred, const StorageT *label, const StorageT *weight,
    const StorageT *weightTensor, StorageT *diff) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {
        int l = int(GPUStorage2ComputeT(label[idx]));
        int modM = idx % M;
        int baseID = (idx / M) * C * M + modM;
        int elementID = baseID + l * M;

        if (weight != NULL) {
            scale *= GPUStorage2ComputeT(weight[l]);
        }

        if (weightTensor == NULL) {
            for (int d = 0; d < C; ++d) {
                int k = baseID + d * M;
                diff[k] = GPUCompute2StorageT(GPUStorage2ComputeT(diff[k]) +
                                              scale *
                                              GPUStorage2ComputeT(pred[k]));
            }
            diff[elementID] = GPUCompute2StorageT(
                GPUStorage2ComputeT(diff[elementID]) - scale);
        } else {
            for (int d = 0; d < C; ++d) {
                int k = baseID + d * M;
                diff[k] = GPUCompute2StorageT(GPUStorage2ComputeT(diff[k]) +
                                              scale *
                                              GPUStorage2ComputeT(pred[k]) *
                                              GPUStorage2ComputeT(
                                                  weightTensor[k % wN]));
            }
            diff[elementID] = GPUCompute2StorageT(
                GPUStorage2ComputeT(diff[elementID]) -
                scale * GPUStorage2ComputeT(weightTensor[elementID % wN]));
        }
    }
}

__global__ void Loss_SmoothL1(size_t CUDA_NUM_LOOPS, size_t N,
                              const StorageT *pred, const StorageT *target,
                              const StorageT *weight, StorageT *loss) {
    // diff = f( weight * (pred - target) )
    // f(x) = 0.5 * x^2    if |x| < 1
    //        |x| - 0.5    otherwise
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {

        ComputeT val =
            GPUStorage2ComputeT(pred[idx]) - GPUStorage2ComputeT(target[idx]);
        if (weight != NULL) val *= GPUStorage2ComputeT(weight[idx]);

        ComputeT abs_val = abs(val);
        if (abs_val < 1) {
            loss[idx] = GPUCompute2StorageT(0.5 * val * val);
        } else {
            loss[idx] = GPUCompute2StorageT(abs_val - 0.5);
        }
    }
}

__global__ void Loss_EuclideanSSE(size_t CUDA_NUM_LOOPS, size_t N,
                              const StorageT *pred, const StorageT *target,
                              const StorageT *weight, StorageT *loss) {
    // diff = f( weight * (pred - target) )
    // f(x) = 0.5 * x^2
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {

        ComputeT val =
            GPUStorage2ComputeT(pred[idx]) - GPUStorage2ComputeT(target[idx]);
        if (weight != NULL) val *= GPUStorage2ComputeT(weight[idx]);

        loss[idx] = GPUCompute2StorageT(0.5 * val * val);
    }
}

__global__ void LossGrad_SmoothL1(
    size_t CUDA_NUM_LOOPS, size_t N, ComputeT scale, const StorageT *pred,
    const StorageT *target, const StorageT *weight, StorageT *diff) {
    // diff = scale * f'( weight * (pred - target) )
    // f'(x) = x         if |x| < 1
    //       = sign(x)   otherwise

    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {

        ComputeT val =
            GPUStorage2ComputeT(pred[idx]) - GPUStorage2ComputeT(target[idx]);
        if (weight != NULL) val *= GPUStorage2ComputeT(weight[idx]);

        ComputeT abs_val = abs(val);
        if (abs_val < 1) {
            diff[idx] = GPUCompute2StorageT(
                GPUStorage2ComputeT(diff[idx]) + scale * val);
        } else {
            diff[idx] = GPUCompute2StorageT(GPUStorage2ComputeT(diff[idx]) +
                                            scale * ((ComputeT(0) < val) -
                                                     (val < ComputeT(0))));
        }
    }
}

__global__ void LossGrad_EuclideanSSE(
    size_t CUDA_NUM_LOOPS, size_t N, ComputeT scale, const StorageT *pred,
    const StorageT *target, const StorageT *weight, StorageT *diff) {
    // diff = scale * f'( weight * (pred - target) )
    // f'(x) = x

    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {

        ComputeT val =
            GPUStorage2ComputeT(pred[idx]) - GPUStorage2ComputeT(target[idx]);
        if (weight != NULL) val *= GPUStorage2ComputeT(weight[idx]);

        diff[idx] = GPUCompute2StorageT(GPUStorage2ComputeT(diff[idx]) + scale * val);        
    }
}

__global__ void Loss_Contrastive(
    size_t CUDA_NUM_LOOPS, size_t N, int C, ComputeT margin, const StorageT *a,
    const StorageT *b, const StorageT *y, StorageT *loss) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {
        ComputeT d = 0.0;
        for (int c = 0; c < C; ++c) {
            int i = idx * C + c;
            ComputeT d_i =
                GPUStorage2ComputeT(a[i]) - GPUStorage2ComputeT(b[i]);
            d += d_i * d_i;
        }
        ComputeT y_n = GPUStorage2ComputeT(y[idx]);
        ComputeT p = max(margin - sqrt(d), ComputeT(0));
        loss[idx] = GPUCompute2StorageT(
            ComputeT(0.5) * (y_n * d + (ComputeT(1) - y_n) * p * p));
    }
}

__global__ void LossGrad_Contrastive(
    size_t CUDA_NUM_LOOPS, size_t N, int C, ComputeT margin, ComputeT scale,
    const StorageT *a, const StorageT *b, const StorageT *y, StorageT *a_diff,
    StorageT *b_diff) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) *
                           (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) +
                            size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N, idxBase + CUDA_NUM_LOOPS); ++idx) {
        if ((int) (GPUStorage2ComputeT(y[idx]))) {
            for (int c = 0; c < C; ++c) {
                int i = idx * C + c;
                ComputeT diff_i =
                    GPUStorage2ComputeT(a[i]) - GPUStorage2ComputeT(b[i]);

                ComputeT beta = scale * diff_i;
                a_diff[i] = GPUCompute2StorageT(
                    GPUStorage2ComputeT(a_diff[i]) + beta);
                b_diff[i] = GPUCompute2StorageT(
                    GPUStorage2ComputeT(b_diff[i]) - beta);
            }
        } else {
            ComputeT dist_sq = 0.0;
            for (int c = 0; c < C; ++c) {
                int i = idx * C + c;
                ComputeT diff_i =
                    GPUStorage2ComputeT(a[i]) - GPUStorage2ComputeT(b[i]);
                dist_sq += diff_i * diff_i;
            }
            ComputeT dist = sqrt(dist_sq);
            ComputeT mdist = margin - dist;

            if (mdist > 0.0) {
                for (int c = 0; c < C; ++c) {
                    int i = idx * C + c;
                    ComputeT diff_i =
                        GPUStorage2ComputeT(a[i]) - GPUStorage2ComputeT(b[i]);
                    ComputeT beta =
                        -scale * mdist / (dist + ComputeT(1e-4)) * diff_i;
                    a_diff[i] = GPUCompute2StorageT(
                        GPUStorage2ComputeT(a_diff[i]) + beta);
                    b_diff[i] = GPUCompute2StorageT(
                        GPUStorage2ComputeT(b_diff[i]) - beta);
                }
            }
        }
    }
}


__global__ void Kernel_OpenCV_BGR_image_to_Marvin(size_t CUDA_NUM_LOOPS, size_t N, size_t channels, size_t height, size_t width, const uint8_t* pIn, uint8_t* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        size_t chn = 2 - (idx % 3);
        size_t col = (idx/3)%width;
        size_t row = (idx/(3*width));
        pOut[(chn*height+row)*width+col] = pIn[idx];
    }
}

void OpenCV_BGR_image_to_Marvin(size_t channels, size_t height, size_t width, const uint8_t* pIn, uint8_t* pOut){
    size_t N = channels * height * width;
    Kernel_OpenCV_BGR_image_to_Marvin<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS >>>(CUDA_GET_LOOPS(N), N, channels, height, width, pIn, pOut);
}

__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const half* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(__half2float(pIn[idx])) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(__half2float(pIn[idx])) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const float* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const double* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const uint8_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const uint16_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const uint32_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const uint64_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const int8_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const int16_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const int32_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const int64_t* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const char* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}
__global__ void Kernel_convert_to_StorageT_subtract(size_t CUDA_NUM_LOOPS, size_t N, size_t sizeofitem, const bool* pIn, const StorageT* pMean, StorageT* pOut) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x)); if (idxBase >= N) return;
    if (pMean==NULL) for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ) pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) );
    else for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx )    pOut[idx] = GPUCompute2StorageT( ComputeT(pIn[idx]) - GPUStorage2ComputeT(pMean[idx % sizeofitem]) );
}

__global__ void Kernel_set_one_hot(size_t CUDA_NUM_LOOPS, size_t N, StorageT* GPUdst, size_t idx2hot){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        GPUdst[idx] = GPUCompute2StorageT( ComputeT(idx == idx2hot) );
    }
}

void GPU_set_one_hot(size_t N, StorageT* GPUdst, size_t idx2hot){
    Kernel_set_one_hot<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,GPUdst,idx2hot);
    checkCUDA(__LINE__,cudaGetLastError());
}

__global__ void Kernel_set_value(size_t CUDA_NUM_LOOPS, size_t N, StorageT* GPUdst, StorageT value){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        GPUdst[idx] = value;
    }
}

void GPU_set_value(size_t N, StorageT* GPUdst, StorageT value){
    Kernel_set_value<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,GPUdst,value);
    checkCUDA(__LINE__,cudaGetLastError());
}

void GPU_set_ones(size_t N, StorageT* GPUdst){
    GPU_set_value(N, GPUdst, CPUCompute2StorageT(1));
}

void GPU_set_zeros(size_t N, StorageT* GPUdst){
    GPU_set_value(N, GPUdst, CPUCompute2StorageT(0));
}

__global__ void Kernel_elementwise_multiplication(size_t CUDA_NUM_LOOPS, size_t N, StorageT* GPUdst, const StorageT* GPUsrcA, const StorageT* GPUsrcB){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        GPUdst[idx] = GPUCompute2StorageT( GPUStorage2ComputeT(GPUsrcA[idx]) * GPUStorage2ComputeT(GPUsrcB[idx]));
    }
}

void GPU_elementwise_multiplication(size_t N, StorageT* GPUdst, const StorageT* GPUsrcA, const StorageT* GPUsrcB){
    Kernel_elementwise_multiplication<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,GPUdst,GPUsrcA,GPUsrcB);
    checkCUDA(__LINE__,cudaGetLastError());
}

__global__ void Kernel_elementwise_comparison(size_t CUDA_NUM_LOOPS, size_t N, StorageT* GPUdst, const StorageT* GPUsrcA, const StorageT* GPUsrcB){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        GPUdst[idx] = GPUCompute2StorageT(ComputeT(bool(GPUStorage2ComputeT(GPUdst[idx])) && (GPUStorage2ComputeT(GPUsrcA[idx]) == GPUStorage2ComputeT(GPUsrcB[idx]))));
    }
}

void GPU_elementwise_comparison(size_t N, StorageT* GPUdst, const StorageT* GPUsrcA, const StorageT* GPUsrcB){
    Kernel_elementwise_comparison<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,GPUdst,GPUsrcA,GPUsrcB);
    //checkCUDA(__LINE__,cudaGetLastError());
}

__global__ void Kernel_copyGPUforward(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* in, StorageT* out, int sizeofitem_in, int sizeofitem_out, int offset){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        int out_base = idx*sizeofitem_out+offset;
        int in_base = idx*sizeofitem_in;
        for(int i=0;i<sizeofitem_in; ++i){
            out[out_base + i] = in[in_base + i];
        }
    }
}

void copyGPUforward(size_t N, const StorageT* in, StorageT* out, int sizeofitem_in, int sizeofitem_out, int offset){
    Kernel_copyGPUforward<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,in,out,sizeofitem_in,sizeofitem_out,offset);
}


__global__ void Kernel_copyGPUbackward(size_t CUDA_NUM_LOOPS, size_t N, StorageT* in, const StorageT* out, int sizeofitem_in, int sizeofitem_out, int offset){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        int in_base = idx*sizeofitem_in;
        int out_base = idx*sizeofitem_out+offset;
        for(int i=0;i<sizeofitem_in; ++i){
            in[in_base + i] = GPUCompute2StorageT( GPUStorage2ComputeT(in[in_base + i]) + GPUStorage2ComputeT(out[out_base + i]) );
        }
    }
}

void copyGPUbackward(size_t N, StorageT* in, const StorageT* out, int sizeofitem_in, int sizeofitem_out, int offset){
    Kernel_copyGPUbackward<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,in,out,sizeofitem_in,sizeofitem_out,offset);
}

__global__ void Kernel_elementwise_acc(size_t CUDA_NUM_LOOPS, size_t N, StorageT* GPUdst, const StorageT* GPUsrc){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        GPUdst[idx] = GPUCompute2StorageT( GPUStorage2ComputeT(GPUdst[idx]) + GPUStorage2ComputeT(GPUsrc[idx]) );
    }
}

__global__ void Kernel_ROIforward_2D(size_t CUDA_NUM_LOOPS, size_t N, StorageT* out, const StorageT* in, const StorageT* start, int od1, int od2, int od3, int id1, int id2, int id3){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t o = idxBase; o < min(N,idxBase+CUDA_NUM_LOOPS); ++o ){
        int n  = (o / (od1*od2*od3));
        int o1 = (o / (    od2*od3)) % od1;
        int o2 = (o /          od3 ) % od2;
        int o3 = (o                ) % od3;
        int i1 = o1 + ((int)(GPUStorage2ComputeT(start[n*3+0])));
        int i2 = o2 + ((int)(GPUStorage2ComputeT(start[n*3+1])));
        int i3 = o3 + ((int)(GPUStorage2ComputeT(start[n*3+2])));
        int i = i3 + ( i2 + ( i1 + n * id1 ) * id2 ) * id3;
        out[o] = in[i];
    }
}

__global__ void Kernel_ROIforward_3D(size_t CUDA_NUM_LOOPS, size_t N, StorageT* out, const StorageT* in, const StorageT* start, int od1, int od2, int od3, int od4, int id1, int id2, int id3, int id4){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t o = idxBase; o < min(N,idxBase+CUDA_NUM_LOOPS); ++o ){
        int n  = (o / (od1*od2*od3*od4));
        int o1 = (o / (    od2*od3*od4)) % od1;
        int o2 = (o / (        od3*od4)) % od2;
        int o3 = (o / (            od4)) % od3;
        int o4 = (o                    ) % od4;
        int i1 = o1 + ((int)(GPUStorage2ComputeT(start[n*4+0])));
        int i2 = o2 + ((int)(GPUStorage2ComputeT(start[n*4+1])));
        int i3 = o3 + ((int)(GPUStorage2ComputeT(start[n*4+2])));
        int i4 = o4 + ((int)(GPUStorage2ComputeT(start[n*4+3])));
        int i = i4 + (i3 + ( i2 + ( i1 + n * id1 ) * id2 ) * id3 ) * id4;
        out[o] = in[i];
    }
}

__global__ void Kernel_ROIforward_4D(size_t CUDA_NUM_LOOPS, size_t N, StorageT* out, const StorageT* in, const StorageT* start, int od1, int od2, int od3, int od4, int od5, int id1, int id2, int id3, int id4, int id5){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t o = idxBase; o < min(N,idxBase+CUDA_NUM_LOOPS); ++o ){
        int n  = (o / (od1*od2*od3*od4*od5));
        int o1 = (o / (    od2*od3*od4*od5)) % od1;
        int o2 = (o / (        od3*od4*od5)) % od2;
        int o3 = (o / (            od4*od5)) % od3;
        int o4 = (o / (                od5)) % od4;
        int o5 = (o                        ) % od5;
        int i1 = o1 + ((int)(GPUStorage2ComputeT(start[n*5+0])));
        int i2 = o2 + ((int)(GPUStorage2ComputeT(start[n*5+1])));
        int i3 = o3 + ((int)(GPUStorage2ComputeT(start[n*5+2])));
        int i4 = o4 + ((int)(GPUStorage2ComputeT(start[n*5+3])));
        int i5 = o5 + ((int)(GPUStorage2ComputeT(start[n*5+4])));
        int i = i5 + (i4 + (i3 + ( i2 + ( i1 + n * id1 ) * id2 ) * id3 ) * id4) * id5;
        out[o] = in[i];
    }
}

__global__ void Kernel_ROIbackward_2D(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* out, StorageT* in, const StorageT* start, int od1, int od2, int od3, int id1, int id2, int id3){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t o = idxBase; o < min(N,idxBase+CUDA_NUM_LOOPS); ++o ){
        int n  = (o / (od1*od2*od3));
        int o1 = (o / (    od2*od3)) % od1;
        int o2 = (o /          od3 ) % od2;
        int o3 = (o                ) % od3;
        int i1 = o1 + ((int)(GPUStorage2ComputeT(start[n*3+0])));
        int i2 = o2 + ((int)(GPUStorage2ComputeT(start[n*3+1])));
        int i3 = o3 + ((int)(GPUStorage2ComputeT(start[n*3+2])));
        int i = i3 + ( i2 + ( i1 + n * id1 ) * id2 ) * id3;
        in[i] = GPUCompute2StorageT( GPUStorage2ComputeT(in[i]) + GPUStorage2ComputeT(out[o]) );
    }
}

__global__ void Kernel_ROIbackward_3D(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* out, StorageT* in, const StorageT* start, int od1, int od2, int od3, int od4, int id1, int id2, int id3, int id4){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t o = idxBase; o < min(N,idxBase+CUDA_NUM_LOOPS); ++o ){
        int n  = (o / (od1*od2*od3*od4));
        int o1 = (o / (    od2*od3*od4)) % od1;
        int o2 = (o / (        od3*od4)) % od2;
        int o3 = (o / (            od4)) % od3;
        int o4 = (o                    ) % od4;
        int i1 = o1 + ((int)(GPUStorage2ComputeT(start[n*4+0])));
        int i2 = o2 + ((int)(GPUStorage2ComputeT(start[n*4+1])));
        int i3 = o3 + ((int)(GPUStorage2ComputeT(start[n*4+2])));
        int i4 = o4 + ((int)(GPUStorage2ComputeT(start[n*4+3])));
        int i = i4 + (i3 + ( i2 + ( i1 + n * id1 ) * id2 ) * id3 ) * id4;
        in[i] = GPUCompute2StorageT( GPUStorage2ComputeT(in[i]) + GPUStorage2ComputeT(out[o]) );
    }
}

__global__ void Kernel_ROIbackward_4D(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* out, StorageT* in, const StorageT* start, int od1, int od2, int od3, int od4, int od5, int id1, int id2, int id3, int id4, int id5){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t o = idxBase; o < min(N,idxBase+CUDA_NUM_LOOPS); ++o ){
        int n  = (o / (od1*od2*od3*od4*od5));
        int o1 = (o / (    od2*od3*od4*od5)) % od1;
        int o2 = (o / (        od3*od4*od5)) % od2;
        int o3 = (o / (            od4*od5)) % od3;
        int o4 = (o / (                od5)) % od4;
        int o5 = (o                        ) % od5;
        int i1 = o1 + ((int)(GPUStorage2ComputeT(start[n*5+0])));
        int i2 = o2 + ((int)(GPUStorage2ComputeT(start[n*5+1])));
        int i3 = o3 + ((int)(GPUStorage2ComputeT(start[n*5+2])));
        int i4 = o4 + ((int)(GPUStorage2ComputeT(start[n*5+3])));
        int i5 = o5 + ((int)(GPUStorage2ComputeT(start[n*5+4])));
        int i = i5 + (i4 + (i3 + ( i2 + ( i1 + n * id1 ) * id2 ) * id3 ) * id4) * id5;
        in[i] = GPUCompute2StorageT( GPUStorage2ComputeT(in[i]) + GPUStorage2ComputeT(out[o]) );
    }
}

__global__ void CoeffElementWiseSumReplace(size_t CUDA_NUM_LOOPS, size_t N, const ComputeT coeff, const StorageT* coeff_data, const size_t num_offset, const size_t dim, const StorageT* in, StorageT* out) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        const ComputeT final_coeff = coeff_data ? ( GPUStorage2ComputeT(coeff_data[num_offset + idx / dim]) * coeff) : coeff;
        out[idx] = GPUCompute2StorageT( GPUStorage2ComputeT(in[idx]) * final_coeff );
    }
}

__global__ void CoeffElementWiseSumAccumulate(size_t CUDA_NUM_LOOPS, size_t N, const ComputeT coeff, const StorageT* coeff_data, const size_t num_offset, const size_t dim, const StorageT* in, StorageT* out) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        const ComputeT final_coeff = coeff_data ? ( GPUStorage2ComputeT(coeff_data[num_offset + idx / dim]) * coeff) : coeff;
        out[idx] = GPUCompute2StorageT(GPUStorage2ComputeT(out[idx]) + GPUStorage2ComputeT(in[idx]) * final_coeff );
    }
}


/* ----------------------------------------------------------------------------
 * The following four functions are inspired by Ross Girshick's Fast-RCNN code,
 * which is copyrighted by Microsoft under an MIT License.
 *
 * Project page: https://github.com/rbgirshick/fast-rcnn
 * License page: https://github.com/rbgirshick/fast-rcnn/blob/master/LICENSE
 * ----------------------------------------------------------------------------
 */
__global__ void Kernel_ROIPoolForward_2D(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* in_data, const StorageT* in_rois, StorageT* out_data, size_t* argmax_data, const ComputeT spatial_scale, const int channels, const int height, const int width, const int pooled_height, const int pooled_width){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;

    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){
        // (n, c, ph, pw) is an element in the pooled output
        int pw = (index) % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int  c = (index / pooled_width / pooled_height) % channels;
        int  n = (index / pooled_width / pooled_height / channels);

        int roi_5n = n*5;
        int roi_batch_ind = GPUStorage2ComputeT(in_rois[roi_5n+0]);
        int roi_start_h = ::round(GPUStorage2ComputeT(in_rois[roi_5n+1]) * spatial_scale);
        int roi_end_h = ::round(GPUStorage2ComputeT(in_rois[roi_5n+2]) * spatial_scale);
        int roi_start_w = ::round(GPUStorage2ComputeT(in_rois[roi_5n+3]) * spatial_scale);
        int roi_end_w = ::round(GPUStorage2ComputeT(in_rois[roi_5n+4]) * spatial_scale);

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);
        ComputeT bin_size_h = static_cast<ComputeT>(roi_height) / static_cast<ComputeT>(pooled_height);
        ComputeT bin_size_w = static_cast<ComputeT>(roi_width) / static_cast<ComputeT>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<ComputeT>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<ComputeT>(pw) * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<ComputeT>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<ComputeT>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        ComputeT maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        size_t maxidx = SIZE_MAX;

        size_t in_offset = (roi_batch_ind * channels + c) * height * width;

        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                size_t in_index = in_offset + h * width + w;
                ComputeT v = GPUStorage2ComputeT(in_data[in_index]);
                if (v > maxval) {
                    maxval = v;
                    maxidx = in_index;
                }
            }
        }
        out_data[index] = GPUCompute2StorageT(maxval);
        if (argmax_data!=NULL)  argmax_data[index] = maxidx;

    }
}

__global__ void Kernel_ROIPoolForward_3D(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* in_data, const StorageT* in_rois, StorageT* out_data, size_t* argmax_data, const ComputeT spatial_scale, const int channels, const int depth, const int height, const int width, const int pooled_depth, const int pooled_height, const int pooled_width){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){

        // (n, c, pd, ph, pw) is an element in the pooled output
        int pw = (index) % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int pd = (index / pooled_width / pooled_height) % pooled_depth;
        int  c = (index / pooled_width / pooled_height / pooled_depth ) % channels;
        int  n = (index / pooled_width / pooled_height / pooled_depth / channels);

        int roi_7n = n * 7;
        int roi_batch_ind = GPUStorage2ComputeT(in_rois[roi_7n+0]);
        int roi_start_d = ::round(GPUStorage2ComputeT(in_rois[roi_7n+1]) * spatial_scale);
        int roi_end_d = ::round(GPUStorage2ComputeT(in_rois[roi_7n+2]) * spatial_scale);
        int roi_start_h = ::round(GPUStorage2ComputeT(in_rois[roi_7n+3]) * spatial_scale);
        int roi_end_h = ::round(GPUStorage2ComputeT(in_rois[roi_7n+4]) * spatial_scale);
        int roi_start_w = ::round(GPUStorage2ComputeT(in_rois[roi_7n+5]) * spatial_scale);
        int roi_end_w = ::round(GPUStorage2ComputeT(in_rois[roi_7n+6]) * spatial_scale);


        // Force malformed ROIs to be 1x1
        int roi_depth = max(roi_end_d - roi_start_d + 1, 1);
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);

        ComputeT bin_size_d = static_cast<ComputeT>(roi_depth) / static_cast<ComputeT>(pooled_depth);
        ComputeT bin_size_h = static_cast<ComputeT>(roi_height) / static_cast<ComputeT>(pooled_height);
        ComputeT bin_size_w = static_cast<ComputeT>(roi_width) / static_cast<ComputeT>(pooled_width);


        int dstart = static_cast<int>(floor(static_cast<ComputeT>(pd) * bin_size_d));
        int hstart = static_cast<int>(floor(static_cast<ComputeT>(ph) * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<ComputeT>(pw) * bin_size_w));
        int dend = static_cast<int>(ceil(static_cast<ComputeT>(pd + 1) * bin_size_d));
        int hend = static_cast<int>(ceil(static_cast<ComputeT>(ph + 1) * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<ComputeT>(pw + 1) * bin_size_w));

        // Add roi offsets and clip to input boundaries

        dstart = min(max(dstart + roi_start_d, 0), depth);
        dend = min(max(dend + roi_start_d, 0), depth);
        hstart = min(max(hstart + roi_start_h, 0), height);
        hend = min(max(hend + roi_start_h, 0), height);
        wstart = min(max(wstart + roi_start_w, 0), width);
        wend = min(max(wend + roi_start_w, 0), width);
        bool is_empty =  (dend <= dstart) || (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        ComputeT maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        size_t maxidx = SIZE_MAX;
        size_t in_offset = (roi_batch_ind * channels + c) * depth * height * width;

        for (int d = dstart; d < dend; ++d) {
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    size_t in_index = in_offset + d * height * width + h * width + w;
                    ComputeT v = GPUStorage2ComputeT(in_data[in_index]);
                    if (v > maxval) {
                        maxval = v;
                        maxidx = in_index;
                    }
                }
            }
        }
        out_data[index] = GPUCompute2StorageT(maxval);
        if (argmax_data!=NULL)  argmax_data[index] = maxidx;
    }
}

__global__ void Kernel_ROIPoolBackward_2D(size_t CUDA_NUM_LOOPS, size_t N, StorageT* in_diff, const StorageT* in_rois, const StorageT* out_diff, const size_t* argmax_data, const ComputeT spatial_scale, const int num_rois, const int channels, const int height, const int width, const int pooled_height, const int pooled_width) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){

        // (n, c, h, w) coords in in data
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;

        ComputeT gradient = GPUStorage2ComputeT(in_diff[index]);
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            int roi_5n = roi_n*5;
            int roi_batch_ind = (int)(GPUStorage2ComputeT(in_rois[roi_5n+0]));
            // Skip if ROI's batch index doesn't match n
            if (n != roi_batch_ind) {
                continue;
            }

            int roi_start_h = ::round(GPUStorage2ComputeT(in_rois[roi_5n+1]) * spatial_scale);
            int roi_end_h = ::round(GPUStorage2ComputeT(in_rois[roi_5n+2]) * spatial_scale);
            int roi_start_w = ::round(GPUStorage2ComputeT(in_rois[roi_5n+3]) * spatial_scale);
            int roi_end_w = ::round(GPUStorage2ComputeT(in_rois[roi_5n+4]) * spatial_scale);

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w && h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
                continue;
            }

            size_t offset = (roi_n * channels + c) * pooled_height * pooled_width;

            // Compute feasible set of pooled units that could have pooled
            // this in unit

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);

            ComputeT bin_size_h = static_cast<ComputeT>(roi_height) / static_cast<ComputeT>(pooled_height);
            ComputeT bin_size_w = static_cast<ComputeT>(roi_width) / static_cast<ComputeT>(pooled_width);

            int phstart = floor(static_cast<ComputeT>(h - roi_start_h) / bin_size_h);
            int phend   =  ceil(static_cast<ComputeT>(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor(static_cast<ComputeT>(w - roi_start_w) / bin_size_w);
            int pwend   =  ceil(static_cast<ComputeT>(w - roi_start_w + 1) / bin_size_w);

            phstart = min(max(phstart, 0), pooled_height);
            phend = min(max(phend, 0), pooled_height);
            pwstart = min(max(pwstart, 0), pooled_width);
            pwend = min(max(pwend, 0), pooled_width);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    size_t out_index = ph * pooled_width + pw;
                    if (argmax_data[offset + out_index] == (h * width + w)) {
                        gradient += GPUStorage2ComputeT(out_diff[offset + out_index]);
                    }
                }
            }
        }
        in_diff[index] = GPUCompute2StorageT(gradient);
    }
}

__global__ void Kernel_ROIPoolBackward_3D(size_t CUDA_NUM_LOOPS, size_t N, StorageT* in_diff, const StorageT* in_rois, const StorageT* out_diff, const size_t* argmax_data, const ComputeT spatial_scale, const int num_rois, const int channels, const int depth, const int height, const int width, const int pooled_depth, const int pooled_height, const int pooled_width) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){

        // (n, c, h, w) coords in in data
        int w = index % width;
        int h = (index / width) % height;
        int d = (index / width / height) % depth;
        int c = (index / width / height / depth) % channels;
        int n = index / width / height / depth / channels;

        ComputeT gradient = GPUStorage2ComputeT(in_diff[index]);
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
            int roi_7n = roi_n*7;
            int roi_batch_ind = (int)(GPUStorage2ComputeT(in_rois[roi_7n+0]));
            // Skip if ROI's batch index doesn't match n
            if (n != roi_batch_ind) {
                continue;
            }

            int roi_start_d = ::round(GPUStorage2ComputeT(in_rois[roi_7n+1]) * spatial_scale);
            int roi_end_d = ::round(GPUStorage2ComputeT(in_rois[roi_7n+2]) * spatial_scale);
            int roi_start_h = ::round(GPUStorage2ComputeT(in_rois[roi_7n+3]) * spatial_scale);
            int roi_end_h = ::round(GPUStorage2ComputeT(in_rois[roi_7n+4]) * spatial_scale);
            int roi_start_w = ::round(GPUStorage2ComputeT(in_rois[roi_7n+5]) * spatial_scale);
            int roi_end_w = ::round(GPUStorage2ComputeT(in_rois[roi_7n+6]) * spatial_scale);

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w && h >= roi_start_h && h <= roi_end_h && d >= roi_start_d && d <= roi_end_d);
            if (!in_roi) {
                continue;
            }

            size_t offset = (roi_n * channels + c) * pooled_depth * pooled_height * pooled_width;

            // Compute feasible set of pooled units that could have pooled
            // this in unit

            // Force malformed ROIs to be 1x1
            int roi_width = max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = max(roi_end_h - roi_start_h + 1, 1);
            int roi_depth = max(roi_end_d - roi_start_d + 1, 1);

            ComputeT bin_size_d = static_cast<ComputeT>(roi_depth) / static_cast<ComputeT>(pooled_depth);
            ComputeT bin_size_h = static_cast<ComputeT>(roi_height) / static_cast<ComputeT>(pooled_height);
            ComputeT bin_size_w = static_cast<ComputeT>(roi_width) / static_cast<ComputeT>(pooled_width);

            int pdstart = floor(static_cast<ComputeT>(d - roi_start_d) / bin_size_d);
            int pdend   =  ceil(static_cast<ComputeT>(d - roi_start_d + 1) / bin_size_d);
            int phstart = floor(static_cast<ComputeT>(h - roi_start_h) / bin_size_h);
            int phend   =  ceil(static_cast<ComputeT>(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor(static_cast<ComputeT>(w - roi_start_w) / bin_size_w);
            int pwend   =  ceil(static_cast<ComputeT>(w - roi_start_w + 1) / bin_size_w);

            pdstart = min(max(pdstart, 0), pooled_depth);
            pdend = min(max(pdend, 0), pooled_depth);
            phstart = min(max(phstart, 0), pooled_height);
            phend = min(max(phend, 0), pooled_height);
            pwstart = min(max(pwstart, 0), pooled_width);
            pwend = min(max(pwend, 0), pooled_width);

            for (int pd = pdstart; pd < pdend; ++pd) {
                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        size_t out_index = (pd * pooled_height + ph) * pooled_width + pw;
                        if (argmax_data[offset + out_index] == ((d * height + h) * width + w)) {
                            gradient += GPUStorage2ComputeT(out_diff[offset+out_index]);
                        }
                    }
                }
            }
        }
        in_diff[index] = GPUCompute2StorageT(gradient);
    }
}

__global__ void Kernel_bsa2b(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* a, StorageT* b){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        b[idx] = GPUCompute2StorageT(GPUStorage2ComputeT(b[idx]) - GPUStorage2ComputeT(a[idx]));
    }
}

void bsa2b(size_t N, const StorageT* a, StorageT* b){
    Kernel_bsa2b<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,a,b);
}

__global__ void Kernel_update_SGDL1(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        ComputeT h  = GPUStorage2ComputeT(gradients[idx]);
        ComputeT g;
        if (w>0)        g = decay;
        else if (w<0)   g = -decay;
        else            g = 0;
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        gradients[idx] = GPUCompute2StorageT(momentum * h + lr * g);
    }
}

__global__ void Kernel_update_SGDL2(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        ComputeT h  = GPUStorage2ComputeT(gradients[idx]);
        ComputeT g  = decay * w;     // L2 regularization
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        gradients[idx] = GPUCompute2StorageT(momentum * h + lr * g);
    }
}

__global__ void Kernel_update_AdaDeltaL1(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        size_t h2_idx = N*(nNets+2)+idx;
        ComputeT h2  = GPUStorage2ComputeT(gradients[h2_idx]);
        
        ComputeT g;
        if (w>0)        g = decay;
        else if (w<0)   g = -decay;
        else            g = 0;
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);

        h = momentum * h + (1-momentum)*g*g;
        g = g * sqrt( (delta+h2) / (delta+h) );
        h2= momentum * h2+ (1-momentum)*g*g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[h2_idx] = GPUCompute2StorageT(h2);
        gradients[idx] = GPUCompute2StorageT(lr * g);
    }
}

__global__ void Kernel_update_AdaDeltaL2(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        size_t h2_idx = N*(nNets+2)+idx;
        ComputeT h2  = GPUStorage2ComputeT(gradients[h2_idx]);

        ComputeT g  = decay * w;     // L2 regularization
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);

        h = momentum * h + (1-momentum)*g*g;
        g = g * sqrt( (delta+h2) / (delta+h) );
        h2= momentum * h2+ (1-momentum)*g*g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[h2_idx] = GPUCompute2StorageT(h2);
        gradients[idx] = GPUCompute2StorageT(lr * g);
    }
}

__global__ void Kernel_update_AdaGradL1(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        ComputeT u  = GPUStorage2ComputeT(gradients[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        ComputeT g;
        if (w>0)        g = decay;
        else if (w<0)   g = -decay;
        else            g = 0;
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        h = g * g + h;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[idx] = GPUCompute2StorageT(momentum * u + lr * g / (sqrt(h) + delta));
    }
}

__global__ void Kernel_update_AdaGradL2(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        ComputeT u  = GPUStorage2ComputeT(gradients[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        ComputeT g  = decay * w;     // L2 regularization
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        h = g * g + h;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[idx] = GPUCompute2StorageT(momentum * u + lr * g / (sqrt(h) + delta));
    }
}

__global__ void Kernel_update_AdamL1(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT momentum2, ComputeT delta, int iter, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        size_t h2_idx = N*(nNets+2)+idx;
        ComputeT h2  = GPUStorage2ComputeT(gradients[h2_idx]);
        ComputeT g;
        if (w>0)        g = decay;
        else if (w<0)   g = -decay;
        else            g = 0;
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        h = momentum * h + (1-momentum )*g;
        h2= momentum2* h2+ (1-momentum2)*g*g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[h2_idx] = GPUCompute2StorageT(h2);
        gradients[idx] = GPUCompute2StorageT(lr * sqrt(1-pow(momentum2,iter)) / (1-pow(momentum,iter)) * h/ (sqrt(h2) + delta));
    }
}

__global__ void Kernel_update_AdamL2(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT momentum2, ComputeT delta, int iter, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        size_t h2_idx = N*(nNets+2)+idx;
        ComputeT h2  = GPUStorage2ComputeT(gradients[h2_idx]);
        ComputeT g  = decay * w;     // L2 regularization
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        h = momentum * h + (1-momentum )*g;
        h2= momentum2* h2+ (1-momentum2)*g*g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[h2_idx] = GPUCompute2StorageT(h2);
        gradients[idx] = GPUCompute2StorageT(lr * sqrt(1-pow(momentum2,iter)) / (1-pow(momentum,iter)) * h/ (sqrt(h2) + delta));
    }
}

__global__ void Kernel_update_NAGL1(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        ComputeT g;
        if (w>0)        g = decay;
        else if (w<0)   g = -decay;
        else            g = 0;
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        ComputeT t  = h;
        h = momentum * h + lr * g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[idx] = GPUCompute2StorageT((1+momentum) * h - momentum * t);
    }
}

__global__ void Kernel_update_NAGL2(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        ComputeT g  = decay * w;     // L2 regularization
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        ComputeT t  = h;
        h = momentum * h + lr * g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[idx] = GPUCompute2StorageT((1+momentum) * h - momentum * t);
    }
}

__global__ void Kernel_update_RMSpropL1(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT rms_decay, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        ComputeT g;
        if (w>0)        g = decay;
        else if (w<0)   g = -decay;
        else            g = 0;
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        h = rms_decay * h + (1-rms_decay) * g * g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[idx] = GPUCompute2StorageT(lr * g / (sqrt(h) + delta));
    }
}

__global__ void Kernel_update_RMSpropL2(size_t CUDA_NUM_LOOPS, size_t N, int nNets, ComputeT decay, ComputeT rms_decay, ComputeT delta, ComputeT lr, const StorageT* weights, StorageT* gradients){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        ComputeT w  = GPUStorage2ComputeT(weights[idx]);
        size_t h_idx = N*(nNets+1)+idx;
        ComputeT h  = GPUStorage2ComputeT(gradients[h_idx]);
        ComputeT g  = decay * w;     // L2 regularization
        for (int k=1; k<nNets+1; ++k) g += GPUStorage2ComputeT(gradients[N*k+idx]);
        h = rms_decay * h + (1-rms_decay) * g * g;
        gradients[h_idx] = GPUCompute2StorageT(h);
        gradients[idx] = GPUCompute2StorageT(lr * g / (sqrt(h) + delta));
    }
}

void update_solver(SolverAlgorithm solver, Regularizer regularizer, int iter, size_t N, int nNets, ComputeT decay, ComputeT momentum, ComputeT momentum2, ComputeT delta, ComputeT rms_decay, ComputeT lr, const StorageT* weights, StorageT* gradients){
    switch (solver){
        case SGD:
            if (regularizer==L1)
                Kernel_update_SGDL1<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,lr,weights,gradients);
            else
                Kernel_update_SGDL2<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,lr,weights,gradients);
            break;
        case AdaDelta:
            if (regularizer==L1)
                Kernel_update_AdaDeltaL1<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,delta,lr,weights,gradients);
            else
                Kernel_update_AdaDeltaL2<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,delta,lr,weights,gradients);
            break;
        case AdaGrad:
            if (regularizer==L1)
                Kernel_update_AdaGradL1<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,delta,lr,weights,gradients);
            else
                Kernel_update_AdaGradL2<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,delta,lr,weights,gradients);
            break;
        case Adam:
            if (regularizer==L1)
                Kernel_update_AdamL1<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,momentum2,delta,iter+1,lr,weights,gradients);
            else
                Kernel_update_AdamL2<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,momentum2,delta,iter+1,lr,weights,gradients);
            break;
        case NAG:
            if (regularizer==L1)
                Kernel_update_NAGL1<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,delta,lr,weights,gradients);
            else
                Kernel_update_NAGL2<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,momentum,delta,lr,weights,gradients);
            break;
        case RMSprop:
            if (regularizer==L1)
                Kernel_update_RMSpropL1<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,rms_decay,delta,lr,weights,gradients);
            else
                Kernel_update_RMSpropL2<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,nNets,decay,rms_decay,delta,lr,weights,gradients);
            break;
    }
    checkCUDA(__LINE__,cudaGetLastError());
}

__global__ void Kernel_xpy(size_t CUDA_NUM_LOOPS, size_t N, const StorageT* x, StorageT* y){
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t idx = idxBase; idx < min(N,idxBase+CUDA_NUM_LOOPS); ++idx ){
        y[idx] = GPUCompute2StorageT( GPUStorage2ComputeT(y[idx]) + GPUStorage2ComputeT(x[idx]));
    }
}

void xpy(size_t N, const StorageT* x, StorageT* y){
    Kernel_xpy<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N),N,x,y);
    checkCUDA(__LINE__,cudaGetLastError());
}

__global__ void Kernel_maxElement(size_t N, const StorageT *x, size_t* pMaxID, ComputeT* pMaxValue){
    const size_t idx = CUDA_NUM_THREADS * blockIdx.x + threadIdx.x;
    if (idx > 0) return;
    //printf("%d %f\n", 0, GPUStorage2ComputeT(x[0]) );
    ComputeT maxValue = GPUStorage2ComputeT(x[0]);
    size_t maxID = 0;
    for (size_t i=1;i<N;++i){
        if (GPUStorage2ComputeT(x[i])>maxValue){
            maxValue = GPUStorage2ComputeT(x[i]);
            maxID = i;
        }
        //printf("%d %f %d\n", i, GPUStorage2ComputeT(x[i]), maxID);
    }
    if (pMaxID!=NULL)    *pMaxID = maxID;
    if (pMaxValue!=NULL) *pMaxValue = maxValue;
}

void GPU_maxElement(size_t N, const StorageT *x, size_t* cpuMaxID, ComputeT* cpuMaxValue){
    size_t* gpuMaxID;    cudaMalloc(&gpuMaxID,    sizeof(size_t));
    ComputeT* gpuMaxValue; cudaMalloc(&gpuMaxValue, sizeof(ComputeT));

    Kernel_maxElement<<<1,1>>>(N, x, gpuMaxID, gpuMaxValue);

    cudaMemcpy(cpuMaxID, gpuMaxID, sizeof(size_t), cudaMemcpyDeviceToHost);          cudaFree(gpuMaxID);
    cudaMemcpy(cpuMaxValue, gpuMaxValue, sizeof(ComputeT), cudaMemcpyDeviceToHost);    cudaFree(gpuMaxValue);
}

__global__ void Kernel_Hasum(size_t N, const half *x, int incx, float *result){
    const int i = CUDA_NUM_THREADS * blockIdx.x + threadIdx.x;
    if (i > 0) return;

    float r = 0;
    for (int i=0;i<N;++i){
        r += fabsf( __half2float(x[i*incx]) );
    }
    *result = r;
}

cublasStatus_t Hasum(cublasHandle_t handle, int n, const half *x, int incx, float *result){
    float* answer;
    cudaMalloc(&answer, sizeof(float));
    Kernel_Hasum<<<1,1>>>(n, x, incx, answer);
    cudaMemcpy(result, answer, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(answer);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t Hgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const half *A, int lda, const half *B, int ldb, const float *beta,  half *C, int ldc){
#if CUDA_VERSION >= 8000
    return cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, beta, C, CUDA_R_16F, ldc);
#else
    return cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, CUBLAS_DATA_HALF, lda, B, CUBLAS_DATA_HALF, ldb, beta, C, CUBLAS_DATA_HALF, ldc);
#endif
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// File format
//////////////////////////////////////////////////////////////////////////////////////////////////

uint8_t typeID(std::type_index t){
    if (t==typeid(half))        return uint8_t(0);
    if (t==typeid(float))       return uint8_t(1);
    if (t==typeid(double))      return uint8_t(2);
    if (t==typeid(uint8_t))     return uint8_t(3);
    if (t==typeid(uint16_t))    return uint8_t(4);
    if (t==typeid(uint32_t))    return uint8_t(5);
    if (t==typeid(uint64_t))    return uint8_t(6);
    if (t==typeid(int8_t))      return uint8_t(7);
    if (t==typeid(int16_t))     return uint8_t(8);
    if (t==typeid(int32_t))     return uint8_t(9);
    if (t==typeid(int64_t))     return uint8_t(10);
    if (t==typeid(char))        return uint8_t(11);
    if (t==typeid(bool))        return uint8_t(12);
    FatalError(__LINE__);       return uint8_t(255);
}

uint8_t readTypeID(std::string filename){
    FILE* fp = fopen(filename.c_str(),"rb");
    while (fp==NULL) {
        std::cerr<<"readTypeID: fail to open file "<<filename<<". Please provide it first. Will retry after 5 seconds."<<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        fp = fopen(filename.c_str(),"rb");
    }
    size_t read_cnt;
    uint8_t fpTypeid; read_cnt = fread((void*)(&fpTypeid), sizeof(uint8_t), 1, fp);     if (read_cnt!=1) { std::cerr<<"Error at readTypeID: no data type. "<<std::endl; FatalError(__LINE__); }
    fclose(fp);
    return fpTypeid;
}

template <class T>
class Tensor{
public:
    std::vector<int> dim;
    T* CPUmem;
    std::string name;

    // compile will check if your time is not correct for writeGPU and readGPU
    void writeGPU(T* GPUmem){
        cudaMemcpy(GPUmem, CPUmem, numel()*sizeof(T), cudaMemcpyHostToDevice);
    };

    void readGPU(T* GPUmem){
        cudaMemcpy(CPUmem, GPUmem, numel()*sizeof(T), cudaMemcpyDeviceToHost);
    };

    Tensor(): CPUmem(NULL){};

    size_t numel(){ return marvin::numel(dim); };

    size_t numBytes(){ return sizeof(T)*numel(); };

    int numofitems(){ return dim[0]; };

    size_t sizeofitem(){ return marvin::sizeofitem(dim); };

    ~Tensor(){
        if (CPUmem!=NULL)   delete[] CPUmem;
    };

    void initialize(T val){
        for (size_t i=0;i<numel();++i){
            CPUmem[i]=val;
        }
    };

    size_t readHeader(FILE* fp){
        size_t read_cnt;
        uint8_t myTypeid = typeID(typeid(T));
        uint32_t myTypesizeof = uint32_t(sizeof(T));
        uint8_t fpTypeid;       read_cnt = fread((void*)(&fpTypeid), sizeof(uint8_t), 1, fp);       if (read_cnt!=1) { std::cerr<<"Error at Tensor::readHeader: no data type. "<<std::endl; FatalError(__LINE__); }
        uint32_t fpTypesizeof;  read_cnt = fread((void*)(&fpTypesizeof), sizeof(uint32_t), 1, fp);  if (read_cnt!=1) { std::cerr<<"Error at Tensor::readHeader: no data size. "<<std::endl; FatalError(__LINE__); }
        int lenName;
        read_cnt = fread((void*)(&lenName), sizeof(int), 1, fp);
        if (read_cnt!=1) { std::cerr<<"Error at Tensor::readHeader: wrong data type. "<<std::endl; FatalError(__LINE__); }
        name.resize(lenName);
        if (lenName>0){
            read_cnt = fread((void*)(name.data()), sizeof(char), lenName, fp);
            if (read_cnt!=lenName) { std::cerr<<"Error at Tensor::readHeader: wrong data type. "<<std::endl; FatalError(__LINE__); }
        }
        int nbDims;
        read_cnt = fread((void*)(&nbDims), sizeof(int), 1, fp);
        if (read_cnt!=1) { std::cerr<<"Error at Tensor::readHeader: wrong data type. "<<std::endl; FatalError(__LINE__); }
        dim.resize(nbDims);
        if (nbDims>0){
            read_cnt = fread((void*)(&dim[0]), sizeof(int), nbDims, fp);
            if (read_cnt!=nbDims) { std::cerr<<"Error at Tensor::readHeader: wrong data type. "<<std::endl; FatalError(__LINE__); }
        }

        size_t headerBytes = sizeof(uint8_t) + sizeof(uint32_t) + sizeof(int) + lenName*sizeof(char) + sizeof(int) + nbDims*sizeof(int);

        if (myTypeid!=fpTypeid || myTypesizeof!=fpTypesizeof){
            std::cerr<<"Error at Tensor::readHeader: wrong data type. "<<std::endl; FatalError(__LINE__);
        }

        return headerBytes;
    };

    //support continuous read across many NdTensors
    T* read(FILE* fp,int batch_size=1){
        if (CPUmem!=NULL){
            delete[] CPUmem;
            CPUmem = NULL;
        }

        size_t read_cnt;

        uint8_t myTypeid = typeID(typeid(T));
        uint32_t myTypesizeof = uint32_t(sizeof(T));

        uint8_t fpTypeid;       read_cnt = fread((void*)(&fpTypeid), sizeof(uint8_t), 1, fp);       if (read_cnt!=1) return NULL;
        uint32_t fpTypesizeof;  read_cnt = fread((void*)(&fpTypesizeof), sizeof(uint32_t), 1, fp);  if (read_cnt!=1) return NULL;

        if (myTypeid!=fpTypeid || myTypesizeof!=fpTypesizeof){

            if (myTypeid==fpTypeid && myTypesizeof!=fpTypesizeof){ std::cerr<<"Tensor read error: same type but different sizeof, maybe different computer architecture. "<<std::endl; FatalError(__LINE__);}

            //if (myTypeid!=fpTypeid){ std::cerr<<"Tensor read error: different types. "<<std::endl; FatalError(__LINE__); }

            if (myTypeid==typeID(typeid(half)) && fpTypeid==typeID(typeid(float))){
                //std::cout<<std::endl<<"converting from float to half"<<std::endl;
                fseek(fp, -(sizeof(uint8_t)+sizeof(uint32_t)), SEEK_CUR);
                Tensor<float>* floatTensor = new Tensor<float>(fp);
                this->dim  = floatTensor->dim ;
                this->name = floatTensor->name;
                Malloc(batch_size);
                for(size_t i=0; i<numel(); ++i){
                    half v = cpu_float2half(floatTensor->CPUmem[i]);
                    memcpy(((half*)(CPUmem))+i,&v,sizeof(half));
                }
                delete floatTensor;
            }else if (myTypeid==typeID(typeid(float)) && fpTypeid==typeID(typeid(half))){
                fseek(fp, -(sizeof(uint8_t)+sizeof(uint32_t)), SEEK_CUR);
                Tensor<half>* halfTensor = new Tensor<half>(fp);
                this->dim  = halfTensor->dim ;
                this->name = halfTensor->name;
                Malloc(batch_size);
                for(size_t i=0; i<numel(); ++i){
                    float v = cpu_half2float(halfTensor->CPUmem[i]);
                    memcpy(((float*)(CPUmem))+i,&v,sizeof(float));
                }
                delete halfTensor;
            }else if (myTypeid==typeID(typeid(double)) && fpTypeid==typeID(typeid(float))){
                fseek(fp, -(sizeof(uint8_t)+sizeof(uint32_t)), SEEK_CUR);
                Tensor<float>* floatTensor = new Tensor<float>(fp);
                this->dim  = floatTensor->dim ;
                this->name = floatTensor->name;
                Malloc(batch_size);
                for(size_t i=0; i<numel(); ++i){
                    double v = double(floatTensor->CPUmem[i]);
                    memcpy(((double*)(CPUmem))+i,&v,sizeof(double));
                }
                delete floatTensor;
            }else if (myTypeid==typeID(typeid(float)) && fpTypeid==typeID(typeid(double))){
                fseek(fp, -(sizeof(uint8_t)+sizeof(uint32_t)), SEEK_CUR);
                Tensor<double>* doubleTensor = new Tensor<double>(fp);
                this->dim  = doubleTensor->dim ;
                this->name = doubleTensor->name;
                Malloc(batch_size);
                for(size_t i=0; i<numel(); ++i){
                    float v = float(doubleTensor->CPUmem[i]);
                    memcpy(((float*)(CPUmem))+i,&v,sizeof(float));
                }
                delete doubleTensor;
            }else if (myTypeid==typeID(typeid(half)) && fpTypeid==typeID(typeid(double))){
                fseek(fp, -(sizeof(uint8_t)+sizeof(uint32_t)), SEEK_CUR);
                Tensor<double>* doubleTensor = new Tensor<double>(fp);
                this->dim  = doubleTensor->dim ;
                this->name = doubleTensor->name;
                Malloc(batch_size);
                for(size_t i=0; i<numel(); ++i){
                    half v = cpu_float2half(float(doubleTensor->CPUmem[i]));
                    memcpy(((half*)(CPUmem))+i,&v,sizeof(half));
                }
                delete doubleTensor;
            }else if (myTypeid==typeID(typeid(float)) && fpTypeid==typeID(typeid(half))){
                fseek(fp, -(sizeof(uint8_t)+sizeof(uint32_t)), SEEK_CUR);
                Tensor<half>* halfTensor = new Tensor<half>(fp);
                this->dim  = halfTensor->dim ;
                this->name = halfTensor->name;
                Malloc(batch_size);
                for(size_t i=0; i<numel(); ++i){
                    double v = double(cpu_half2float(halfTensor->CPUmem[i]));
                    memcpy(((double*)(CPUmem))+i,&v,sizeof(double));
                }
                delete halfTensor;
            }else{
                std::cerr<<"Tensor conversion is not supported: from Type "<<fpTypeid<<" to Type "<<myTypeid<<std::endl;
                FatalError(__LINE__);
            }

        }else{
            int lenName;
            read_cnt = fread((void*)(&lenName), sizeof(int), 1, fp);
            if (read_cnt!=1) return NULL;
            name.resize(lenName);
            if (lenName>0){
                read_cnt = fread((void*)(name.data()), sizeof(char), lenName, fp);
                if (read_cnt!=lenName) return NULL;
            }
            int nbDims;
            read_cnt = fread((void*)(&nbDims), sizeof(int), 1, fp);
            if (read_cnt!=1) return NULL;
            dim.resize(nbDims);
            if (nbDims>0){
                read_cnt = fread((void*)(&dim[0]), sizeof(int), nbDims, fp);
                if (read_cnt!=nbDims) return NULL;
            }

            size_t n = numel();
            Malloc(batch_size);
            read_cnt = fread((void*)(CPUmem), sizeof(T), n, fp);
            if (read_cnt!=n){
                delete [] CPUmem;
                CPUmem = NULL;
                return NULL;
            }
        }

        return CPUmem;
    };

    void Malloc(int batch_size){
        size_t n = numel();
        std::cout<<"  ";        memorySizePrint(n*sizeof(T));   std::cout<<std::endl;

        if (batch_size==1 || dim[0]%batch_size ==0){
            CPUmem = new T [n];
        }else{
            int dim0 =  (dim[0]/batch_size + 1) * batch_size;
            size_t oversize = n/dim[0] * dim0;
            CPUmem = new T [oversize];
            memset((void*)(CPUmem+n),0, (oversize-n)*sizeof(T));
        }
    };

    T* read(std::string filename,int batch_size=1){
        FILE* fp = fopen(filename.c_str(),"rb");
        while (fp==NULL) {
            std::cerr<<"Tensor:read: fail to open file "<<filename<<". Please provide it first. Will retry after 5 seconds."<<std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            fp = fopen(filename.c_str(),"rb");
        }
        read(fp,batch_size);
        fclose(fp);
        return CPUmem;
    };

    //write without header
    void writeHeader(FILE* fp, std::vector<int> dim2write){
        uint8_t myTypeid = typeID(typeid(T));
        fwrite((void*)(&myTypeid), sizeof(uint8_t), 1, fp);
        uint32_t typesizeof = uint32_t(sizeof(T));
        fwrite((void*)(&typesizeof), sizeof(uint32_t), 1, fp);
        int lenName = name.size();
        fwrite((void*)(&lenName), sizeof(int), 1, fp);
        if (lenName>0) fwrite((void*)(name.data()), sizeof(char), lenName, fp);
        int nbDims = dim2write.size();
        fwrite((void*)(&nbDims), sizeof(int), 1, fp);
        if (nbDims>0) fwrite((void*)(&dim2write[0]), sizeof(int), nbDims, fp);
        if (ferror (fp)){
            std::cerr << "disk writing failed"<<std::endl;
            FatalError();
        }
    };

    void writeData(FILE* fp, size_t max_size = 0){
        size_t n = numel();
        if (max_size !=0 ) n = min(n,max_size);
        if (n>0){
            fwrite((void*)(CPUmem), sizeof(T), n, fp);
            if (ferror (fp)){
                std::cerr << "disk writing failed" << std::endl;
                FatalError();
            }
        }
    };

    //support continuous write across many NdTensors
    //write with header
    void write(FILE* fp){
        writeHeader(fp,dim);
        writeData(fp);
    };

    void write(std::string filename){
        FILE* fp = fopen(filename.c_str(),"wb");
        while (fp==NULL) {
            std::cerr<<"Tensor::write: fail to open file "<<filename<<". Will retry after 5 seconds."<<std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            fp = fopen(filename.c_str(),"wb");
        }
        write(fp);
        fclose(fp);
        return;
    };

    Tensor(std::string filename, int batch_size=1): CPUmem(NULL){ read(filename,batch_size); };

    Tensor(FILE* fp): CPUmem(NULL){ read(fp); };

    Tensor(std::vector<int> dim_): dim(dim_){ CPUmem = new T [numel()]; };

    Tensor(std::vector<int> dim_, T* ptr_data): dim(dim_){ CPUmem = ptr_data; };

    Tensor(std::vector<int> dim_, T initValue): dim(dim_){
        int n = numel();
        CPUmem = new T [n];
        if (initValue == T(0))
            memset(CPUmem, 0, n*sizeof(T));
        else
            for (int i=0;i<n;++i) CPUmem[i] = initValue;

    };

    Tensor(std::string name_, std::vector<int> dim_): name(name_),dim(dim_){
        CPUmem = new T [numel()];
    };

    void permute(std::vector<size_t> v){
        size_t nbItems = numofitems();
        size_t sizeofitem_ = sizeofitem();
        size_t nbBytes = sizeofitem_ * sizeof(T);
        T* CPUmemNew = new T[numel()];
        memcpy(CPUmemNew, CPUmem, nbItems * nbBytes);
        for (size_t i=0;i<nbItems;++i){
            memcpy(CPUmem+i*sizeofitem_, CPUmemNew+v[i]*sizeofitem_, nbBytes);
        }
        delete [] CPUmemNew;
    };


    void printRange(){
        int n = numel();
        if (n==0){
            std::cout<<"Emtpy tensor"<<std::endl;
            return;
        }
        T maxValue = CPUmem[0];
        T minValue = CPUmem[0];

        for (int i=0;i<n;++i){
            if (maxValue<CPUmem[i])     maxValue=CPUmem[i];
            if (CPUmem[i]<minValue)     minValue=CPUmem[i];
        }
        std::cout<< "Value Range ["<<minValue<<", "<<maxValue<<"]"<<std::endl;
    };

    void print(std::vector<int> display_dim){

        std::cout<<"  name:"<<name<<" dim"; veciPrint(dim); std::cout<<std::endl;
        switch (display_dim.size()){
            case 1:
                for (int i=0;i<min((size_t)(display_dim[0]),numel());++i)
                    std::cout<<CPUmem[i]<<" ";
                std::cout<<std::endl;
                break;
            case 2:
                for (int i=0;i<display_dim[0];++i){
                    for (int j=0;j<display_dim[1];++j){
                        std::cout<<(CPUmem[i*dim[display_dim.size()-1]+j])<<" ";
                    }
                    std::cout<<std::endl;
                }
                break;
            case 3:
                for (int i=0;i<display_dim[0];++i){
                    for (int j=0;j<display_dim[1];++j){
                        for (int k=0;k<display_dim[2];++k){
                            std::cout<<CPUmem[i*dim[dim.size()-2]*dim[dim.size()-1]+j*dim[dim.size()-1]+k]<<" ";
                        }
                        std::cout<<std::endl;
                    }
                    std::cout<<std::endl;
                }
                break;
        }

    };
};

template <class T>
std::vector<Tensor<T>*> readTensors(std::string filename, size_t max_count = SIZE_MAX){

    FILE* fp = fopen(filename.c_str(),"rb");

    while (fp==NULL) {
        std::cerr<<"readTensors: fail to open file "<<filename<<". Please provide it first. Will retry after 5 seconds."<<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        fp = fopen(filename.c_str(),"rb");
    }

    std::vector<Tensor<T>*> tensors;
    size_t count = 0;
    while (feof(fp)==0) {
        tensors.push_back(new Tensor<T>(fp));
        count++;
        if (count>=max_count) break;
        int c = getc(fp);
        ungetc(c, fp);
    }
    fclose(fp);
    return tensors;
}

template <class T>
void writeTensors(std::string filename, std::vector<Tensor<T>*> tensors){
    FILE* fp = fopen(filename.c_str(),"wb");
    while (fp==NULL) {
        std::cerr<<"writeTensors: fail to open file "<<filename<<". Disk full? Will retry after 5 seconds."<<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        fp = fopen(filename.c_str(),"wb");
    }

    for(int i=0;i<tensors.size();++i){
        tensors[i]->write(fp);
    }
    fclose(fp);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
// Response and Layer
//////////////////////////////////////////////////////////////////////////////////////////////////

class Response{
public:
    std::string name;
    cudnnTensorDescriptor_t desc;
    cublasHandle_t cublasHandle;
    std::vector<cudnnTensorDescriptor_t> desc_group;
    std::vector<int> number_group;

    bool isProxy;

    StorageT* dataGPU;
    StorageT* diffGPU;
    bool need_diff;
    std::vector<int> dim;
    std::vector<int> stride;

    std::vector<ComputeT> receptive_field;
    std::vector<ComputeT> receptive_gap;
    std::vector<ComputeT> receptive_offset;

    size_t sizeofitem(){ return marvin::sizeofitem(dim); };

    size_t numBytes(){ return sizeofStorageT*(marvin::numel(dim)); };

    Response(std::string name_, bool need_diff_=false): name(name_), dataGPU(NULL), diffGPU(NULL), need_diff(need_diff_), isProxy(false){
        checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&desc));
    };

    size_t Malloc(std::vector<int> dim_, StorageT* dataGPUexisting=NULL, StorageT* diffGPUexisting=NULL){
        size_t memoryBytes = 0;
        if (dataGPU==NULL){ // two layers (one for training, one for testing) may output to the same response and Malloc twice, ignore the second time

            dim = dim_;
            stride.resize(dim.size());

            stride[dim.size()-1] = 1;
            for (int d=dim.size()-2;d>=0;--d){
                stride[d] = stride[d+1] *  dim[d+1];
            }

            std::cout << dim.size() << std::endl;

            checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(desc,
                                                    CUDNNStorageT,
                                                    dim.size(),
                                                    &dim[0],
                                                    &stride[0]) );

            std::cout<<"                                                                               ";
            std::cout<< (need_diff? "* " : "  ");

            std::cout<<name; veciPrint(dim);
            if (!receptive_field.empty())   {   std::cout<<" RF"; vecfPrint(receptive_field);   }
            if (!receptive_gap.empty())     {   std::cout<<" GP"; vecfPrint(receptive_gap);     }
            if (!receptive_offset.empty())  {   std::cout<<" OF"; vecfPrint(receptive_offset);      }

            std::cout<<std::endl;

            if (dataGPUexisting==NULL){            
                checkCUDA(__LINE__, cudaMalloc(&dataGPU, numel(dim) * sizeofStorageT) );
                memoryBytes += numel(dim) * sizeofStorageT;
            }else{
                dataGPU = dataGPUexisting;
                isProxy = true;
            }

            if (need_diff){
                if (diffGPUexisting==NULL){                
                    checkCUDA(__LINE__, cudaMalloc(&diffGPU, numel(dim) * sizeofStorageT) );
                    memoryBytes += numel(dim) * sizeofStorageT;
                }else{
                    diffGPU = diffGPUexisting;
                    isProxy = true;
                }
            }
        }else{
            if (!same_dim(dim, dim_)){

                std::cerr<<std::endl<<"Response["<< name <<"] Malloc dimension mis-matched: ";
                veciPrint(dim);
                std::cerr<<" vs ";
                veciPrint(dim_);
                std::cerr<<std::endl;

                if (numel(dim)!=numel(dim_)) FatalError(__LINE__);
            }
        }
        return memoryBytes;
    };


    cudnnTensorDescriptor_t getDesc(int group=1){ // must be called after malloc
        if (group==1){
            return desc;
        }else{
            for(int i=0;i<number_group.size();++i){
                if (number_group[i]==group){
                    return desc_group[i];
                }
            }
        }
        number_group.push_back(group);
        cudnnTensorDescriptor_t desc_new;
        checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&desc_new));
        std::vector<int> dim_new = dim;
        dim_new[1] = dim[1]/group;
        checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(desc_new,
                                                CUDNNStorageT,
                                                dim_new.size(),
                                                &dim_new[0],
                                                &stride[0]) );
        desc_group.push_back(desc_new);
        return desc_new;
    }

    ~Response(){
        checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(desc));
        for (int i=0; i<desc_group.size();++i){
            checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(desc_group[i]));
        }
        if (dataGPU!=NULL && !isProxy) checkCUDA(__LINE__, cudaFree(dataGPU));
        if (diffGPU!=NULL && !isProxy) checkCUDA(__LINE__, cudaFree(diffGPU));
    };

    void clearDiff(){
        if (diffGPU!=NULL && !isProxy){
            checkCUDA(__LINE__, cudaMemset(diffGPU, 0, sizeofStorageT * numel(dim)));
        }
    };

    void print(std::vector<int> display_dim, bool printData=true){
        if (!printData && diffGPU==NULL) return;
        Tensor<StorageT>* feature = new Tensor<StorageT>(dim);
        feature->readGPU((printData? dataGPU: diffGPU));
        feature->print(display_dim);
        delete feature;
    };


    int checkNaN(){
        return marvin::checkNaN(dataGPU, numel(dim));
    };

    int checkNaNdiff(){
        return marvin::checkNaN(diffGPU, numel(dim));
    };

    ComputeT ameanData(){
        if (dataGPU!=NULL){
            ComputeT result;
            size_t n = numel(dim);
            //std::cout<<"n="<<n<<std::endl;
            //std::cout<<"cublasHandle="<<cublasHandle<<std::endl;
            //std::cout<<"dataGPU="<<dataGPU<<std::endl;
            checkCUBLAS(__LINE__, GPUasum(cublasHandle, n, dataGPU, 1, &result));
            result /= ComputeT(n);
            return result;
        }else{
            return -1;
        }
    };
    ComputeT ameanDiff(){
        if (diffGPU!=NULL){
            ComputeT result;
            size_t n = numel(dim);
            checkCUBLAS(__LINE__, GPUasum(cublasHandle, n, diffGPU, 1, &result));
            result /= ComputeT(n);
            return result;
        }else{
            return -1;
        }
    };
};

class Layer {
public:
    StorageT *weight_dataGPU;
    StorageT *weight_diffGPU;
    StorageT *weight_histGPU;

    StorageT *bias_dataGPU;
    StorageT *bias_diffGPU;
    StorageT *bias_histGPU;

    std::vector<Response *> in;
    std::vector<Response *> out;

    std::mt19937 rng;
    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    // parameters:
    int GPU;

    std::string name;
    Phase phase;
    bool train_me; // user specify whether they want to tune this layer

    ComputeT weight_lr_mult;
    Filler weight_filler;
    ComputeT weight_filler_param;
    std::vector<int> weight_dim;
    size_t weight_numel;
    ComputeT weight_decay_mult;

    ComputeT bias_lr_mult;
    Filler bias_filler;
    ComputeT bias_filler_param;
    std::vector<int> bias_dim;
    size_t bias_numel;
    ComputeT bias_decay_mult;

    std::vector<Layer*> sub_layers;

    Layer() : phase(TrainingTesting), train_me(false), weight_dataGPU(NULL),
              weight_diffGPU(NULL), weight_histGPU(NULL), bias_dataGPU(NULL),
              bias_diffGPU(NULL), bias_histGPU(NULL), weight_numel(0),
              bias_numel(0), weight_decay_mult(ComputeT(1)),
              bias_decay_mult(ComputeT(1)) {
        checkCUDNN(__LINE__, cudnnCreate(&cudnnHandle));
        checkCUBLAS(__LINE__, cublasCreate(&cublasHandle));
        std::random_device rd;
        rng.seed(rd());
    };

    Layer(std::string name_) : name(name_), phase(TrainingTesting),
                               train_me(false), weight_dataGPU(NULL),
                               weight_diffGPU(NULL), weight_histGPU(NULL),
                               bias_dataGPU(NULL), bias_diffGPU(NULL),
                               bias_histGPU(NULL), weight_numel(0),
                               bias_numel(0), weight_decay_mult(ComputeT(1)),
                               bias_decay_mult(ComputeT(1)) {
        checkCUDNN(__LINE__, cudnnCreate(&cudnnHandle));
        checkCUBLAS(__LINE__, cublasCreate(&cublasHandle));
        std::random_device rd;
        rng.seed(rd());
    };

    virtual ~Layer() {
        if (weight_dataGPU != NULL)
            checkCUDA(__LINE__, cudaFree(weight_dataGPU));

        if (bias_dataGPU != NULL) checkCUDA(__LINE__, cudaFree(bias_dataGPU));
    };

    ComputeT ameanWeightData() {
        if (weight_dataGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(weight_dim);
        checkCUBLAS(__LINE__,
                    GPUasum(cublasHandle, n, weight_dataGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    ComputeT ameanWeightDiff() {
        if (weight_diffGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(weight_dim);
        checkCUBLAS(__LINE__,
                    GPUasum(cublasHandle, n, weight_diffGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    int checkNaNWeight(){
        return marvin::checkNaN(weight_dataGPU, numel(weight_dim));
    };

    int checkNaNWeightDiff(){
        return marvin::checkNaN(weight_diffGPU, numel(weight_dim));
    };    

    ComputeT ameanBiasData() {
        if (bias_dataGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(bias_dim);
        checkCUBLAS(__LINE__,
                    GPUasum(cublasHandle, n, bias_dataGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    ComputeT ameanBiasDiff() {
        if (bias_diffGPU == NULL) return -1;
        ComputeT result;
        size_t n = numel(bias_dim);
        checkCUBLAS(__LINE__,
                    GPUasum(cublasHandle, n, bias_diffGPU, 1, &result));
        result /= ComputeT(n);
        return result;
    };

    int checkNaNBias(){
        return marvin::checkNaN(bias_dataGPU, numel(bias_dim));
    };

    int checkNaNBiasDiff(){
        return marvin::checkNaN(bias_diffGPU, numel(bias_dim));
    };       

    void addIn(Response *r) { in.push_back(r); };

    void addOut(Response *r) { out.push_back(r); };

    virtual size_t Malloc(Phase phase_) {    // by default, do nothing
        std::cout << (train_me ? "* " : "  ");
        std::cout << name << std::endl;
        return 0;
    };

    virtual void forward(Phase phase_) { };  // by default, do nothing
    virtual void backward(Phase phase_) { }; // by default, do nothing
    virtual void display() { };

    virtual bool isDataLayer() { return false; };

    void fillGPU(StorageT *GPUmem, std::vector<int> dim, Filler filler,
                 ComputeT param = 0) {
        int n = numel(dim);
        StorageT *CPUbuf = new StorageT[n];
        switch (filler) {
            case Xavier: {
                int fan_in = ComputeT(n / dim[0]);
                ComputeT scale = sqrt(ComputeT(3) / fan_in);

                //default_random_engine generator;
                std::uniform_real_distribution<ComputeT> distribution(-scale,
                                                                      scale);
                for (StorageT *p = CPUbuf; p != CPUbuf + n; ++p) {
                    *p = CPUCompute2StorageT(distribution(rng));
                }
            }
            break;
            case Gaussian: {
                std::normal_distribution<ComputeT> distribution(0, param);
                for (StorageT *p = CPUbuf; p != CPUbuf + n; ++p) {
                    *p = CPUCompute2StorageT(distribution(rng));
                }
            }
            break;
            case Constant: {
                StorageT paramStorageT = CPUCompute2StorageT(param);
                for (StorageT *p = CPUbuf; p != CPUbuf + n; ++p) {
                    *p = paramStorageT;
                }
            }
            break;
        }
        checkCUDA(__LINE__, cudaMemcpy(GPUmem, CPUbuf, n * sizeofStorageT,
                                       cudaMemcpyHostToDevice));

        delete[] CPUbuf;
    }

    void randInit() {
        if (weight_dataGPU != NULL) fillGPU(weight_dataGPU, weight_dim, weight_filler, weight_filler_param);
        if (bias_dataGPU != NULL) fillGPU(bias_dataGPU, bias_dim, bias_filler, bias_filler_param);
        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->randInit();
    };

    void clearDiff() {
        if (weight_diffGPU != NULL)
            checkCUDA(__LINE__, cudaMemset(weight_diffGPU, 0,
                                           sizeofStorageT * weight_numel));
        if (bias_diffGPU != NULL)
            checkCUDA(__LINE__, cudaMemset(bias_diffGPU, 0,
                                           sizeofStorageT * bias_numel));
        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->clearDiff();
    };

    void clearHist() {
        if (weight_diffGPU != NULL)
            checkCUDA(__LINE__, cudaMemset(weight_histGPU, 0,
                                           sizeofStorageT * weight_numel));
        if (bias_diffGPU != NULL)
            checkCUDA(__LINE__, cudaMemset(bias_histGPU, 0,
                                           sizeofStorageT * bias_numel));
        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->clearHist();
    };

    void setWeights(std::vector<Tensor<StorageT> *> weights) {
        for (int i = 0; i < weights.size(); ++i) {
            if (weight_dataGPU != NULL &&
                weights[i]->name == name + ".weight") {
                if (numel(weight_dim) == numel(weights[i]->dim)) {
                    if (!same_dim(weight_dim, weights[i]->dim)) {
                        std::cout << "[Warning] " << name <<
                        ".weight is loaded with mismatched dimensions ";
                        std::cout << "need";
                        veciPrint(weight_dim);
                        std::cout << " vs. file";
                        veciPrint(weights[i]->dim);
                        std::cout << std::endl;
                    }
                    std::cout << " " << name << ".weight";
                    veciPrint(weights[i]->dim);
                    std::cout << " is set." << std::endl;
                    weights[i]->writeGPU(weight_dataGPU);
                } else {
                    std::cout << "[Warning] " << name <<
                    ".weight is found but not loaded because the numels are mismatched: ";
                    std::cout << "need";
                    veciPrint(weight_dim);
                    std::cout << " vs. file";
                    veciPrint(weights[i]->dim);
                    std::cout << std::endl;
                }
            }
            if (bias_dataGPU != NULL && weights[i]->name == name + ".bias") {
                if (numel(bias_dim) == numel(weights[i]->dim)) {
                    if (!same_dim(bias_dim, weights[i]->dim)) {
                        std::cout << "[Warning] " << name <<
                        ".bias is loaded with mismatched dimensions ";
                        std::cout << "need";
                        veciPrint(bias_dim);
                        std::cout << " vs. file";
                        veciPrint(weights[i]->dim);
                        std::cout << std::endl;
                    }
                    std::cout << " " << name << ".bias";
                    veciPrint(weights[i]->dim);
                    std::cout << " is set." << std::endl;
                    weights[i]->writeGPU(bias_dataGPU);
                } else {
                    std::cout << "[Warning] " << name <<
                    ".bias is found but not loaded because the numels are mismatched: ";
                    std::cout << "need";
                    veciPrint(bias_dim);
                    std::cout << " vs. file";
                    veciPrint(weights[i]->dim);
                    std::cout << std::endl;
                }

            }
        }
        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->setWeights(weights);
    };

    void saveWeights(FILE *fp) {
        if (weight_dataGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".weight", weight_dim);
            t->readGPU(weight_dataGPU);
            t->write(fp);
            delete t;
        }

        if (bias_dataGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".bias", bias_dim);
            t->readGPU(bias_dataGPU);
            t->write(fp);
            delete t;
        }

        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->saveWeights(fp);
    };

    void printWeights(std::vector<int> display_weight,
                      std::vector<int> display_bias) {
        if (weight_dataGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".weight", weight_dim);
            t->readGPU(weight_dataGPU);
            t->print(display_weight);
            delete t;
        }
        if (bias_dataGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".bias", bias_dim);
            t->readGPU(bias_dataGPU);
            t->print(display_bias);
            delete t;
        }

        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->printWeights(display_weight,display_bias);
    };

    void setDiffs(std::vector<Tensor<StorageT> *> weights) {
        for (int i = 0; i < weights.size(); ++i) {
            if (weight_diffGPU != NULL &&
                weights[i]->name == name + ".weight_diff") {
                std::cout << " " << name << ".weight_diff";
                veciPrint(weights[i]->dim);
                std::cout << " is set." << std::endl;
                weights[i]->writeGPU(weight_diffGPU);
            }
            if (bias_diffGPU != NULL &&
                weights[i]->name == name + ".bias_diff") {
                std::cout << " " << name << ".bias_diff";
                veciPrint(weights[i]->dim);
                std::cout << " is set." << std::endl;
                weights[i]->writeGPU(bias_diffGPU);
            }
        }

        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->setDiffs(weights);
    };

    void saveDiffs(FILE *fp) {
        if (weight_diffGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".weight_diff", weight_dim);
            t->readGPU(weight_diffGPU);
            t->write(fp);
            delete t;
        }

        if (bias_diffGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".bias_diff", bias_dim);
            t->readGPU(bias_diffGPU);
            t->write(fp);
            delete t;
        }

        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->saveDiffs(fp);
    };

    void printDiffs(std::vector<int> display_weight,
                    std::vector<int> display_bias) {
        if (weight_diffGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".weight_diff", weight_dim);
            t->readGPU(weight_diffGPU);
            t->print(display_weight);
            delete t;
        }
        if (bias_diffGPU != NULL) {
            Tensor <StorageT> *t = new Tensor<StorageT>(
                name + ".bias_diff", bias_dim);
            t->readGPU(bias_diffGPU);
            t->print(display_bias);
            delete t;
        }
        for(int l=0;l<sub_layers.size();++l) sub_layers[l]->printDiffs(display_weight,display_bias);
    };

    void update() {
        if (train_me) {
            if (weight_numel > 0 && weight_histGPU != NULL)
                bsa2b(weight_numel, weight_histGPU, weight_dataGPU);
            if (bias_numel > 0 && bias_histGPU != NULL)
                bsa2b(bias_numel, bias_histGPU, bias_dataGPU);
            for(int l=0;l<sub_layers.size();++l) sub_layers[l]->update();
        }
    };

};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Layers
//////////////////////////////////////////////////////////////////////////////////////////////////

class DataLayer : public Layer {
public:
    // parameters:
    bool random;
    int counter;
    int epoch;
    bool isDataLayer(){ return true; };
    DataLayer(): counter(0), epoch(0), random(false){};
    DataLayer(std::string name_): Layer(name_), counter(0), epoch(0), random(false){};
    virtual int numofitems() = 0;
    virtual void shuffle() = 0;
};


class ConstantLayer: public DataLayer {
    std::vector<ComputeT> data;
public:
    ConstantLayer(std::string name_): DataLayer(name_){
        train_me = false;
    };
    ConstantLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, data)
        train_me = false;
    };
    ~ConstantLayer(){ };
    int numofitems(){ return 1; };
    void shuffle(){};
    void forward(Phase phase_){ ++epoch; };
    size_t Malloc(Phase phase_){
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;
        if (!in.empty()){   std::cout<<"ConstantLayer shouldn't have any in's"<<std::endl; FatalError(__LINE__); }
        if (out.size()!=1){   std::cout<<"ConstantLayer should have one out"<<std::endl; FatalError(__LINE__); }
        size_t memoryBytes = 0;
        std::vector<int> dim;
        dim.push_back(data.size());
        dim.push_back(1);
        dim.push_back(1);
        out[0]->need_diff = false;
        memoryBytes += out[0]->Malloc(dim);
        StorageT* dataStorageT = new StorageT[data.size()];
        for (size_t i=0; i<data.size(); ++i) dataStorageT[i] = CPUCompute2StorageT(data[i]); 
        checkCUDA(__LINE__, cudaMemcpy(out[0]->dataGPU, dataStorageT, data.size() * sizeofStorageT, cudaMemcpyHostToDevice) );
        delete [] dataStorageT;
        return memoryBytes;
    };
};


class TensorLayer: public DataLayer {
    StorageT* tensorGPU;
public:
    std::vector<std::string> files;
    std::vector<std::vector<int> > dim;

    TensorLayer(std::string name_): DataLayer(name_), tensorGPU(NULL){
        train_me = false;
    };

    TensorLayer(JSON* json): tensorGPU(NULL){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, files        )
        train_me = false;
    };

    ~TensorLayer(){
        if (tensorGPU!=NULL) checkCUDA(__LINE__, cudaFree(tensorGPU));
    };

    int numofitems(){
        return dim[0][0];
    };

    void shuffle(){

    };

    void forward(Phase phase_){
        ++epoch;
    };

    size_t Malloc(Phase phase_){
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (!in.empty()){   std::cout<<"TensorLayer shouldn't have any in's"<<std::endl; FatalError(__LINE__); }
        if (out.empty()){   std::cout<<"TensorLayer should have some out's"<<std::endl; FatalError(__LINE__); }
        if (out.size()!=files.size()){  std::cout<<"TensorLayer: # of out's should match the # of in's"<<std::endl; FatalError(__LINE__); }

        size_t memoryBytes = 0;

        dim.resize(files.size());
        for (size_t i=0;i<files.size();++i){
            Tensor<StorageT>* tensorCPU = new Tensor<StorageT>(files[i]);
            dim[i] = tensorCPU->dim;
            out[i]->need_diff = false;
            std::cout<<"tensorCPU->dim="; veciPrint(tensorCPU->dim); std::cout<<std::endl;
            memoryBytes += out[i]->Malloc(tensorCPU->dim);
            checkCUDA(__LINE__, cudaMemcpy(out[i]->dataGPU, tensorCPU->CPUmem, tensorCPU-> numBytes(), cudaMemcpyHostToDevice) );
            delete tensorCPU;
        }
        return memoryBytes;
    };
};

class SequenceGenerationLayer: public DataLayer {
    int channel;
    size_t iter;
public:
    int length;
    int seed;
    Response* resultResponse;
    std::string result;
    std::string map2char;
    Tensor<char>* tensor2char;

    SequenceGenerationLayer(std::string name_): DataLayer(name_), resultResponse(NULL), tensor2char(NULL), iter(0){
        train_me = false;
    };

    SequenceGenerationLayer(JSON* json): resultResponse(NULL), tensor2char(NULL), iter(0){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, map2char     )
        SetOrDie(json, length       )
        SetOrDie(json, seed         )
        SetOrDie(json, result       )

        train_me = false;
    };

    ~SequenceGenerationLayer(){
        if (tensor2char!=NULL) delete tensor2char;
    };

    int numofitems(){
        return length;
    };

    void shuffle(){};

    void forward(Phase phase_){
        if (iter == 0){
            GPU_set_zeros(1, out[1]->dataGPU);
            GPU_set_one_hot(channel, out[0]->dataGPU, seed);

            //std::cout<<seed<<" "<<std::endl;
            std::cout<<tensor2char->CPUmem[seed];
        }else{
            GPU_set_ones(1, out[1]->dataGPU);

            size_t maxID;
            ComputeT maxValue;
            GPU_maxElement(channel, resultResponse->dataGPU, &maxID, &maxValue);

            //maxID = iter%65;
            //std::cout<<"CPU maxID="<<maxID<<std::endl;

            //FatalError(__LINE__);
            //std::cout<<maxID<<" "<<maxValue<<std::endl;
            std::cout<<tensor2char->CPUmem[maxID];

            GPU_set_one_hot(channel, out[0]->dataGPU, maxID);
        }

        ++iter;

        if (iter ==length)
            ++epoch;
    };

    size_t Malloc(Phase phase_){
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (out.size()!=2){  std::cout<<"SequenceGenerationLayer: # of out's should be 2."<<std::endl; FatalError(__LINE__); }

        size_t memoryBytes = 0;

        tensor2char = new Tensor<char>(map2char);
        channel = tensor2char->numel();

        std::vector<int> dim;
        dim.push_back(1);
        dim.push_back(1);
        dim.push_back(channel);

        out[0]->need_diff = false;
        memoryBytes += out[0]->Malloc(dim);

        out[1]->need_diff = false;
        dim[2]=1;
        memoryBytes += out[1]->Malloc(dim);
        
        return memoryBytes;
    };
};



class MemoryDataLayer : public DataLayer {
    std::vector<Tensor<StorageT>*> dataCPU;
    public:
    std::vector<std::string> file_data;
    std::vector<std::string> file_mean;
    std::vector<ComputeT> scale;
    std::vector<ComputeT> mean;
    int batch_size;

    int numofitems(){
        return dataCPU[0]->dim[0];
    };
    void init(){
        train_me = false;
        std::cout<<"MemoryDataLayer "<<name<<" loading data: "<<std::endl;
        dataCPU.resize(file_data.size());
        for (int i =0;i<file_data.size();i++){
            dataCPU[i] = new Tensor<StorageT> (file_data[i],batch_size);
            dataCPU[i]->print(veci(0));
        }

        if (file_mean.size()>0){
            for (int i =0;i<file_mean.size();i++){
                Tensor<StorageT>* meanCPU = new Tensor<StorageT>(file_mean[i],batch_size);
                meanCPU->print(veci(0));

                if (meanCPU->numel() != dataCPU[i]->sizeofitem()){
                    std::cerr<<"mean tensor file size error: "<<std::endl;
                    std::cerr<<"mean"; veciPrint(meanCPU->dim); std::cerr<<std::endl;
                    std::cerr<<"data"; veciPrint(dataCPU[i]->dim); std::cerr<<std::endl;
                    FatalError(__LINE__);
                };
                StorageT* d  = dataCPU[i]->CPUmem;
                StorageT* dE = dataCPU[i]->CPUmem + dataCPU[i]->numel();

                StorageT* m  = meanCPU->CPUmem;
                StorageT* mE = meanCPU->CPUmem + meanCPU->numel();

                while(d!=dE){
                    *d = CPUCompute2StorageT( CPUStorage2ComputeT(*d) - CPUStorage2ComputeT(*m) );
                    ++m;
                    if (m==mE) m = meanCPU->CPUmem;
                    ++d;
                }
                delete meanCPU;
            }
        }

        for (int i =0;i<scale.size();i++){
            if (scale[i]!=1){
                StorageT* dE = dataCPU[i]->CPUmem + dataCPU[i]->numel();
                for(StorageT* d  = dataCPU[i]->CPUmem; d!=dE; ++d){
                    *d = CPUCompute2StorageT( CPUStorage2ComputeT(*d) * scale[i] );
                }
            }
        }
        for (int i =0;i<mean.size();i++){
            if (mean[i]!=0){
                StorageT* dE = dataCPU[i]->CPUmem + dataCPU[i]->numel();
                for(StorageT* d  = dataCPU[i]->CPUmem; d!=dE; ++d){
                    *d = CPUCompute2StorageT( CPUStorage2ComputeT(*d) - mean[i] );
                }
            }
        }

        if (phase!=Testing) shuffle();
    }

    MemoryDataLayer(std::string name_, Phase phase_, std::vector<std::string> file_data_, int batch_size_): DataLayer(name_), batch_size(batch_size_), file_data(file_data_){
        phase = phase_;
        init();
    };
    MemoryDataLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       Training)
        SetOrDie(json, file_data    )
        SetValue(json, file_mean,   std::vector<std::string>(0))
        SetValue(json, batch_size,  64)
        SetValue(json, scale,       std::vector<ComputeT>(0))
        SetValue(json, mean,        std::vector<ComputeT>(0))
        SetValue(json, random,      true)
        init();
    };
    ~MemoryDataLayer(){
        for (int i =0; i<dataCPU.size();i++){
            delete dataCPU[i];
        }
    };
    size_t Malloc(Phase phase_){
        if (phase == Training && phase_==Testing) return 0;
        
        if (!in.empty()){   std::cout<<"MemoryDataLayer shouldn't have any in's"<<std::endl; FatalError(__LINE__); }
        if (out.empty()){   std::cout<<"MemoryDataLayer should have some out's"<<std::endl; FatalError(__LINE__); }
        if (out.size()!=file_data.size()){  std::cout<<"MemoryDataLayer: # of out's should match the # of in's"<<std::endl; FatalError(__LINE__); }

        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;
        for (int i = 0;i < file_data.size(); i++){
            out[i]->need_diff = false;
            std::vector<int> data_dim = dataCPU[i]->dim;
            data_dim[0] = batch_size;
            out[i]->receptive_field.resize(data_dim.size()-2);  fill_n(out[i]->receptive_field.begin(), data_dim.size()-2,1);
            out[i]->receptive_gap.resize(data_dim.size()-2);    fill_n(out[i]->receptive_gap.begin(),   data_dim.size()-2,1);
            out[i]->receptive_offset.resize(data_dim.size()-2); fill_n(out[i]->receptive_offset.begin(),data_dim.size()-2,0);
            memoryBytes += out[i]->Malloc(data_dim);
        }
        return memoryBytes;
    }
    void shuffle(){
        if (!random) return;
        std::vector<size_t> v = randperm(dataCPU[0]->numofitems(), rng);
        for(int i =0; i <dataCPU.size();i++){
            dataCPU[i]->permute(v);
        }
    };

    void forward(Phase phase_){
        if (counter + batch_size >= dataCPU[0]->numofitems() ){
            ++epoch;
            if(phase!=Testing){
                shuffle();
                counter = 0;
            }
        }
        for(int i =0; i <dataCPU.size();i++){
            checkCUDA(__LINE__, cudaMemcpy(out[i]->dataGPU, dataCPU[i]->CPUmem +  (size_t(counter) * size_t( dataCPU[i]->sizeofitem())), batch_size * dataCPU[i]->sizeofitem() * sizeofStorageT, cudaMemcpyHostToDevice) );     
        }
        counter+=batch_size;
        if (counter >= dataCPU[0]->numofitems()) counter = 0;
    };
};


#if USE_OPENCV
class ImageDataLayer : public DataLayer {
    StorageT* dataCPU;
    StorageT* dataGPU;
    StorageT* labelCPU;
    StorageT* labelGPU;

    Tensor<StorageT>* labelTensor;

    std::vector<size_t> ordering;
    std::vector<std::string> img_fname;

    std::future<void> lock;
    int epoch_prefetch;

public:
    std::string file_list;
    std::string file_label;
    std::vector<int> image_output;
    int batch_size;
    ComputeT mean_value;

    ImageDataLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       Training)
        SetOrDie(json, image_output)
        SetOrDie(json, file_list)
        SetOrDie(json, file_label)
        SetOrDie(json, mean_value)
        SetValue(json, batch_size,  64)
        SetValue(json, random,      true)
        init();
    };

    void shuffle(){
        if (!random) return;
        if (phase!=Testing){
            ordering = randperm(img_fname.size(), rng);
        }
    };     

    int numofitems(){
        return img_fname.size();
    };
    void init(){
        train_me = false;
        std::cout<<"ImageDataLayer "<<name<<" loading data: ";

        std::ifstream fin(file_list);
        while (!fin.eof()){
            std::string fname;
            fin>>fname;
            if (fin.eof()) break;
            img_fname.push_back(fname);
        };
        fin.close();
        std::cout<<"# of images = "<<img_fname.size()<<std::endl;


        labelTensor = new Tensor<StorageT>(file_label);
        labelTensor->print(veci(0));

        if (phase!=Testing){
            shuffle();
        }else{
            ordering.resize(numofitems());
            for (int i=0;i<numofitems();++i) ordering[i]=i;
        }
    }

    ImageDataLayer(std::string name_, Phase phase_, int batch_size_): DataLayer(name_), batch_size(batch_size_){
        phase = phase_;

        dataCPU  =NULL;
        labelCPU =NULL;
        dataGPU  =NULL;
        labelGPU =NULL;

        init();
    };

    ~ImageDataLayer(){
        if (labelTensor!=NULL) delete labelTensor;
        if (dataCPU!=NULL)  delete [] dataCPU;
        if (labelCPU!=NULL) delete [] labelCPU;
        if (dataGPU!=NULL)  checkCUDA(__LINE__, cudaFree(dataGPU));
        if (labelGPU!=NULL) checkCUDA(__LINE__, cudaFree(labelGPU));
    };

    void prefetch(){

        size_t perImageSize = 3*image_output[0]*image_output[1];
        cv::Size resize_size(image_output[0],image_output[1]);

        for (size_t i=0;i<batch_size;++i){
            // choose image
            int image_i = ordering[counter];

            // read image
            cv::Mat image_RGB = cv::imread(img_fname[image_i],CV_LOAD_IMAGE_UNCHANGED);
            cv::Mat image_RGB_resize;

            // resize image
            cv::resize(image_RGB,image_RGB_resize,resize_size);

            // convert image and subtract mean
            StorageT* pImage = dataCPU+i*perImageSize;
            StorageT* pImageEnd   = pImage+perImageSize;
            uchar* pPixel = image_RGB_resize.data;
            while(pImage!=pImageEnd){
                *pImage = CPUCompute2StorageT(ComputeT(*pPixel)- mean_value);
                ++pImage; ++pPixel;
            }

            // copy label
            memcpy(labelCPU+i*labelTensor->sizeofitem(), labelTensor->CPUmem+image_i*labelTensor->sizeofitem(), labelTensor->sizeofitem()*sizeofStorageT );
            
            counter++;
            if (counter>= ordering.size()){
                if (phase!=Testing) shuffle();
                counter = 0;
                ++epoch_prefetch;
            }
        }

        // copy from CPU to GPU
        checkCUDA(__LINE__, cudaMemcpy(dataGPU, dataCPU, batch_size*perImageSize*sizeofStorageT, cudaMemcpyHostToDevice) );
        checkCUDA(__LINE__, cudaMemcpy(labelGPU, labelCPU, batch_size*labelTensor->sizeofitem()*sizeofStorageT, cudaMemcpyHostToDevice) );
    };


    size_t Malloc(Phase phase_){
        if (phase == Training && phase_==Testing) return 0;
        
        if (!in.empty()){   std::cout<<"ImageDataLayer shouldn't have any in's"<<std::endl; FatalError(__LINE__); }
        if (out.empty()){   std::cout<<"ImageDataLayer should have some out's"<<std::endl; FatalError(__LINE__); }
        if (out.size()!=2){  std::cout<<"ImageDataLayer: # of out's should be 2"<<std::endl; FatalError(__LINE__); }

        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        dataCPU  = new StorageT[batch_size*3*image_output[0]*image_output[1]];
        checkCUDA(__LINE__, cudaMalloc(&dataGPU, batch_size*3*image_output[0]*image_output[1]*sizeofStorageT) );
        memoryBytes += batch_size*3*image_output[0]*image_output[1]*sizeofStorageT;

        labelCPU = new StorageT[batch_size* labelTensor->sizeofitem() ];
        checkCUDA(__LINE__, cudaMalloc(&labelGPU, batch_size* labelTensor->sizeofitem()*sizeofStorageT) );
        memoryBytes += batch_size* labelTensor->sizeofitem()*sizeofStorageT;

        out[0]->need_diff = false;
        std::vector<int> data_dim;
        data_dim.push_back(batch_size);
        data_dim.push_back(3);
        data_dim.push_back(image_output[0]);
        data_dim.push_back(image_output[1]);
        out[0]->receptive_field.resize(data_dim.size()-2);  fill_n(out[0]->receptive_field.begin(), data_dim.size()-2,1);
        out[0]->receptive_gap.resize(data_dim.size()-2);    fill_n(out[0]->receptive_gap.begin(),   data_dim.size()-2,1);
        out[0]->receptive_offset.resize(data_dim.size()-2); fill_n(out[0]->receptive_offset.begin(),data_dim.size()-2,0);
        memoryBytes += out[0]->Malloc(data_dim);

        out[1]->need_diff = false;
        std::vector<int> label_dim;
        label_dim.push_back(batch_size);
        label_dim.push_back(labelTensor->dim[1]);
        label_dim.push_back(labelTensor->dim[2]);
        label_dim.push_back(labelTensor->dim[3]);
        out[1]->receptive_field.resize(label_dim.size()-2);  fill_n(out[1]->receptive_field.begin(), label_dim.size()-2,1);
        out[1]->receptive_gap.resize(label_dim.size()-2);    fill_n(out[1]->receptive_gap.begin(),   label_dim.size()-2,1);
        out[1]->receptive_offset.resize(label_dim.size()-2); fill_n(out[1]->receptive_offset.begin(),label_dim.size()-2,0);
        memoryBytes += out[1]->Malloc(label_dim);

        lock = std::async(std::launch::async,&ImageDataLayer::prefetch,this);

        return memoryBytes;
    };

    void forward(Phase phase_){
        lock.wait();
        epoch = epoch_prefetch;

        std::swap(out[0]->dataGPU,dataGPU);
        std::swap(out[1]->dataGPU,labelGPU);
        lock = std::async(std::launch::async,&ImageDataLayer::prefetch,this);
    };
};
#endif


class PlaceHolderDataLayer : public DataLayer {
    public:
    std::vector<int> dim;
    std::string file_mean;
    StorageT* meanGPU;

    int numofitems(){
        return 1;
    };
    void init(){
        train_me = false;
        std::cout<<"PlaceHolderDataLayer "<<name<<std::endl;
    };

    PlaceHolderDataLayer(std::string name_, Phase phase_): DataLayer(name_){
        phase = phase_;
        init();
    };
    PlaceHolderDataLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       Testing)
        SetOrDie(json, dim)
        SetValue(json, file_mean,   "")
        init();
    };
    ~PlaceHolderDataLayer(){
    };
    size_t Malloc(Phase phase_){
        if (phase == Training && phase_==Testing) return 0;
        
        if (!in.empty()){   std::cout<<"PlaceHolderDataLayer shouldn't have any in's"<<std::endl; FatalError(__LINE__); }
        if (out.empty()){   std::cout<<"PlaceHolderDataLayer should have some out's"<<std::endl; FatalError(__LINE__); }

        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        out[0]->need_diff = false;
        out[0]->receptive_field.resize(dim.size()-2);  fill_n(out[0]->receptive_field.begin(), dim.size()-2,1);
        out[0]->receptive_gap.resize(dim.size()-2);    fill_n(out[0]->receptive_gap.begin(),   dim.size()-2,1);
        out[0]->receptive_offset.resize(dim.size()-2); fill_n(out[0]->receptive_offset.begin(),dim.size()-2,0);
        memoryBytes += out[0]->Malloc(dim);

        // mean
        if(file_mean.empty()){
            meanGPU = NULL;
        }else{
            Tensor<StorageT>* meanCPU = new Tensor<StorageT>(file_mean);
            meanCPU->print(veci(0));
            checkCUDA(__LINE__, cudaMalloc(&meanGPU, meanCPU->numBytes()) );
            memoryBytes += meanCPU->numBytes();
            meanCPU->writeGPU(meanGPU);
            delete meanCPU;
        }

        return memoryBytes;
    }
    void shuffle(){};
    void forward(Phase phase_){};
};

template <class T>
class DiskDataLayer : public DataLayer {
    std::future<void> lock;
    
    std::vector<size_t> ordering; 
    std::bernoulli_distribution* distribution_bernoulli;
    std::vector<std::uniform_int_distribution<int>*> distribution_uniform;

    std::vector<FILE*> dataFILE;
    std::vector<T*> dataCPU;
    std::vector<T*> dataGPU;
    std::vector<T*> item_raw;

    Tensor<StorageT>* labelCPUall;
    Tensor<StorageT>* labelCPU;
    std::vector<StorageT*> mean_data_GPU;
    StorageT* labelGPU;

    size_t numel_per_channel_crop ;
    size_t numel_all_channel_crop ;
    size_t numel_per_channel_orgi ; 
    size_t numel_batch_all_channel_crop ;

    int epoch_prefetch;

    size_t bytes_per_item;
    size_t headerBytes;
    std::vector<int> size_data;
    public:
    bool mirror;
    std::vector<int> size_crop;
    std::vector<std::string> file_data;
    std::vector<std::string> file_mean;
    std::string file_label;
    int batch_size;

    int numofitems(){
        return labelCPUall->numofitems();
    };

    void init(){
        epoch_prefetch  = 0;
        distribution_bernoulli = new std::bernoulli_distribution(0.5);  
        //dataFILE = NULL;
        //dataCPU = NULL;
        //dataGPU = NULL;
        labelCPU = NULL;
        labelGPU = NULL;
        labelCPUall = NULL;
        train_me = false;
        std::cout<<"DiskDataLayer "<<name<<" loading data: "<<std::endl;

        dataCPU.resize(file_data.size());
        dataGPU.resize(file_data.size());
        item_raw.resize(file_data.size());
        dataFILE.resize(file_data.size());

        // open data file
        for (int i = 0;i<file_data.size();++i){
            dataFILE[i] = fopen(file_data[i].c_str(),"rb");
            if (dataFILE[i] ==NULL){
                std::cerr<<"Fail to open the data file"<<std::endl;
                FatalError(__LINE__);
            }
        }

        mean_data_GPU.resize(file_mean.size());
        for (int i =0;i<file_mean.size();i++){
            Tensor<StorageT>* meanCPU = new Tensor<StorageT>(file_mean[i],batch_size);
            meanCPU->print(veci(0));
            checkCUDA(__LINE__, cudaMalloc(&mean_data_GPU[i], meanCPU->numBytes()) );
            meanCPU->writeGPU(mean_data_GPU[i]);
            delete meanCPU;
        }

        
        Tensor<T> tensor;
        headerBytes = tensor.readHeader(dataFILE[0]);

        size_data.insert( size_data.end(), tensor.dim.begin()+1, tensor.dim.end() );


        numel_per_channel_crop = numel(size_crop);
        numel_all_channel_crop = size_data[0] * numel_per_channel_crop;
        numel_per_channel_orgi = sizeofitem(size_data);
        numel_batch_all_channel_crop = batch_size*numel_all_channel_crop;
        bytes_per_item = sizeof(T)* numel(size_data);

        std::vector<int> data_dim;
        data_dim.push_back(batch_size);
        data_dim.push_back(size_data[0]);
        data_dim.insert( data_dim.end(), size_crop.begin(), size_crop.end() );

        for (int i = 0;i<file_data.size();++i){
            dataCPU[i]  = new T[numel(data_dim)];
            item_raw[i] = new T[numel(size_data)];
        }


        // for label
        labelCPUall = new Tensor<StorageT>(file_label);
        labelCPUall -> print(veci(0));
        std::cout<<"    "; labelCPUall->printRange();
        while (labelCPUall->dim.size()<size_data.size()+1) labelCPUall->dim.push_back(1);
        std::vector<int> label_dim = labelCPUall->dim;
        label_dim[0] = batch_size;
        labelCPU = new Tensor<StorageT>(label_dim);


        distribution_uniform.resize(size_crop.size());
        for (int d=0; d<size_crop.size(); d++){
            distribution_uniform[d] = new std::uniform_int_distribution<int>(0,size_data[d+1] - size_crop[d]);
        }

        if (phase!=Testing){
            shuffle();
        }else{
            ordering.resize(numofitems());
            for (int i=0;i<numofitems();++i) ordering[i]=i;
        }
    };

    DiskDataLayer(std::string name_, Phase phase_, bool mirror_, std::vector<int> size_data_, std::vector<int> size_crop_, std::vector<std::string> file_data_, std::string file_label_, int batch_size_): 
        DataLayer(name_), mirror(mirror_), size_data(size_data_), size_crop(size_crop_), file_data(file_data_), file_label(file_label_), batch_size(batch_size_){
        phase = phase_;
        init();
    };

    DiskDataLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       Training)
        SetValue(json, mirror,      false)
        SetOrDie(json, file_data    )
        SetValue(json, file_mean,   std::vector<std::string>(0))
        SetValue(json, file_label,"")
        SetOrDie(json, batch_size   )
        SetOrDie(json, size_crop    )
        SetValue(json, random,      true)
        init();
    };

    ~DiskDataLayer(){
        if (lock.valid()) lock.wait();
        delete distribution_bernoulli;
        for (int i=0;i<distribution_uniform.size();++i) delete distribution_uniform[i];
        for (int i = 0; i<dataFILE.size();++i){
            if (dataFILE[i]!=NULL) fclose(dataFILE[i]);
        }
        for (int i = 0; i<dataCPU.size();++i){
            if (dataCPU[i]!=NULL) delete [] dataCPU[i];
        }
        for (int i = 0; i<item_raw.size();++i){
            if (item_raw[i]!=NULL) delete [] item_raw[i];
        }

        if (labelCPU!=NULL) delete labelCPU;
        if (labelCPUall!=NULL) delete labelCPUall;

        for (int i = 0; i<dataGPU.size();++i){
            if (dataGPU[i]!=NULL) checkCUDA(__LINE__, cudaFree(dataGPU[i]));
        }

        if (labelGPU!=NULL) checkCUDA(__LINE__, cudaFree(labelGPU));


        for (int i =0;i<file_mean.size();i++){
            if (mean_data_GPU[i]!=NULL) checkCUDA(__LINE__, cudaFree(mean_data_GPU[i]));
        }
    };


    void shuffle(){
        if (!random) return;
        if (phase!=Testing){
            ordering = randperm(labelCPUall->numofitems(), rng);
        }
    }; 

    void prefetch(){

        checkCUDA(__LINE__,cudaSetDevice(GPU));

        
        std::vector<size_t> begin_coor(size_crop.size());

        for (size_t i=0;i<batch_size;++i){

            int image_i = ordering[counter];
            //std::cout<<"i"<<i<<"image_i"<<image_i<<"bytes_per_item"<<bytes_per_item<<std::endl;

            //label 
            size_t labelSizeOfItem = labelCPU->sizeofitem();
            memcpy(labelCPU->CPUmem+i*labelSizeOfItem, labelCPUall->CPUmem+image_i*labelSizeOfItem, labelSizeOfItem*sizeofStorageT);
            
            // mirror
            bool mirror_this = false;
            if (mirror) mirror_this = ((*distribution_bernoulli)(rng));
            if (numel_per_channel_orgi != numel_per_channel_crop || mirror_this){
                for (int d=0;d<size_crop.size();++d){
                    begin_coor[d] = (numel_per_channel_orgi == numel_per_channel_crop) ? 0 : ((*(distribution_uniform[d]))(rng));
                }
            }
            

            //for (int data_i = 0; data_i<file_data.size();data_i++){
            for (int data_i = 0; data_i<file_data.size();data_i++){
                // read file
                fseek(dataFILE[data_i], headerBytes + bytes_per_item * image_i, SEEK_SET);
                size_t read_cnt = fread(item_raw[data_i], 1, bytes_per_item, dataFILE[data_i]);
                if (read_cnt != bytes_per_item){
                    std::cerr<<"Error reading file for DiskDataLayer::prefetch : "<<dataFILE[data_i]<<std::endl;
                    std::cerr<<"data_i"<<data_i<<"read_cnt: "<<read_cnt<<" bytes_per_item: "<<bytes_per_item<<std::endl;
                    FatalError(__LINE__);
                }

                T* memBegin = dataCPU[data_i] + i * numel_all_channel_crop;
                if (numel_per_channel_orgi == numel_per_channel_crop && !mirror_this){
                    memcpy(memBegin, item_raw[data_i], bytes_per_item);
                }
                else{
                    if (size_crop.size()==2){
                        for (size_t x_crop = 0; x_crop < size_crop[0]; ++ x_crop){
                            size_t x_orgi = x_crop + begin_coor[0];
                            for (size_t y_crop=0; y_crop < size_crop[1]; ++ y_crop){
                                size_t y_orgi = y_crop + begin_coor[1];
                                if (mirror_this) y_orgi = size_data[2] - 1 - y_orgi;

                                size_t idx_orgi = x_orgi * size_data[2] + y_orgi;
                                size_t idx_crop = x_crop * size_crop[1] + y_crop;
                                for (size_t c=0; c<size_data[0];++c){
                                    memBegin[idx_crop+c*numel_per_channel_crop] =  item_raw[data_i][idx_orgi+c*numel_per_channel_orgi];
                                }
                            }               
                        }
                    }else if (size_crop.size()==3){
                        for (size_t x_crop = 0; x_crop < size_crop[0]; ++ x_crop){
                            size_t x_orgi = x_crop + begin_coor[0];
                            for (size_t y_crop=0; y_crop < size_crop[1]; ++ y_crop){
                                size_t y_orgi = y_crop + begin_coor[1];
                                if (mirror_this) y_orgi = size_data[2] - 1 - y_orgi;

                                for (size_t z_crop=0; z_crop<size_crop[2]; ++z_crop){
                                    size_t z_orgi = z_crop + begin_coor[2];

                                    size_t idx_orgi = (x_orgi * size_data[2] + y_orgi) * size_data[3] + z_orgi;
                                    size_t idx_crop = (x_crop * size_crop[1] + y_crop) * size_crop[2] + z_crop;
                                    for (size_t c=0; c<size_data[0];++c){
                                        memBegin[idx_crop+c*numel_per_channel_crop] =  item_raw[data_i][idx_orgi+c*numel_per_channel_orgi];
                                    }
                                }
                            }               
                        }
                    }
                    else{
                        std::cerr<<"Error: dimension unimplemented. You can implement by yourself."<<std::endl;
                        FatalError(__LINE__);
                    }
                }
            }//for (int data_i = 0; data_i<file_data.size();data_i++)

            counter++;
            if (counter>= ordering.size()){
                if (phase!=Testing) shuffle();
                counter = 0;
                ++epoch_prefetch;
            }
        }//end for (size_t i=0;i<batch_size;++i)
        //std::cout<<"numel_batch_all_channel_crop:  "<<numel_batch_all_channel_crop<<std::endl;
        for (int data_i = 0; data_i<file_data.size();data_i++){
            checkCUDA(__LINE__, cudaMemcpy( dataGPU[data_i],  dataCPU[data_i],  numel_batch_all_channel_crop*sizeof(T), cudaMemcpyHostToDevice) );
        }
            
        labelCPU->writeGPU(labelGPU);

    };

    void forward(Phase phase_){
        lock.wait();
        epoch = epoch_prefetch;
        for (int data_i = 0; data_i<file_data.size();data_i++){
            StorageT* mean_data =  (data_i<mean_data_GPU.size()? mean_data_GPU[data_i]: NULL );
            Kernel_convert_to_StorageT_subtract<<<CUDA_GET_BLOCKS(numel_batch_all_channel_crop), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(numel_batch_all_channel_crop), numel_batch_all_channel_crop, numel_all_channel_crop, dataGPU[data_i], mean_data, out[data_i]->dataGPU);
        }
        std::swap(out[file_data.size()]->dataGPU,labelGPU);
        lock = std::async(std::launch::async,&DiskDataLayer<T>::prefetch,this);
    };


    size_t Malloc(Phase phase_){

        if (phase == Training && phase_==Testing) return 0;

        

        if (out.size()!=file_data.size()+1){    std::cout<<"DiskDataLayer: # of out's should match the # of in-1"<<std::endl; FatalError(__LINE__); }
        if (! (in.size()==0 || in.size()<file_data.size())){
            std::cerr<< "DiskDataLayer in.size()==0 || in.size<file_data.size()"<<std::endl;
            FatalError(__LINE__);
        }
        size_t memoryBytes = 0;

        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        std::vector<int> data_dim;
        data_dim.push_back(batch_size);
        data_dim.push_back(size_data[0]);
        data_dim.insert( data_dim.end(), size_crop.begin(), size_crop.end() );

        for (int data_i = 0; data_i<file_data.size();data_i++){
            out[data_i]->need_diff = false;
            out[data_i]->receptive_field.resize(data_dim.size()-2); fill_n(out[data_i]->receptive_field.begin(),data_dim.size()-2,1);
            out[data_i]->receptive_gap.resize(data_dim.size()-2);   fill_n(out[data_i]->receptive_gap.begin(),data_dim.size()-2,1);     
            out[data_i]->receptive_offset.resize(data_dim.size()-2);fill_n(out[data_i]->receptive_offset.begin(),data_dim.size()-2,0);
            memoryBytes += out[data_i]->Malloc(data_dim);
        }

        out[file_data.size()]->need_diff = false;
        memoryBytes += out[file_data.size()]->Malloc(labelCPU->dim);
        checkCUDA(__LINE__, cudaMalloc(&labelGPU, labelCPU->numBytes()) );
        memoryBytes += labelCPU->numBytes();

        for (int data_i = 0; data_i<file_data.size();data_i++){
            checkCUDA(__LINE__, cudaMalloc(&dataGPU[data_i], numel_batch_all_channel_crop * sizeof(T)) );
            memoryBytes += numel_batch_all_channel_crop * sizeof(T);
        }

        lock = std::async(std::launch::async,&DiskDataLayer<T>::prefetch,this);

        return memoryBytes;
    };  
};


class ConvolutionLayer : public Layer {
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    std::vector<StorageT *> fwdAlgoWorkspaces;
    std::vector<StorageT *> bwdDataAlgoWorkspaces;
    std::vector<StorageT *> bwdFilterAlgoWorkspaces;

    std::vector<size_t> fwdAlgoWorkspaceSizes;
    std::vector<size_t> bwdDataAlgoWorkspaceSizes;
    std::vector<size_t> bwdFilterAlgoWorkspaceSizes;
public:
    cudnnConvolutionFwdAlgo_t fwdAlgo;
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;

    int num_output;
    std::vector<int> window;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> upscale;
    int group;

    void init(){
        weight_dim.push_back(num_output);
        weight_dim.push_back(0);  // need the channel size from the input
        weight_dim.insert( weight_dim.end(), window.begin(), window.end() );

        bias_dim.resize(weight_dim.size(), 1);
        bias_dim[1] = num_output;
    };

    ConvolutionLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, train_me,            true)
        SetOrDie(json, num_output           )
        SetOrDie(json, window               )
        SetValue(json, weight_lr_mult,      1.0)
        SetValue(json, weight_filler,       Xavier)
        SetValue(json, weight_filler_param, 0.0)
        SetValue(json, bias_lr_mult,        2.0)
        SetValue(json, bias_filler,         Constant)
        SetValue(json, bias_filler_param,   0.0)
        SetValue(json, weight_decay_mult,   1.0)
        SetValue(json, bias_decay_mult,     1.0)
        SetValue(json, group,               1)

        std::vector<int> ones  = std::vector<int>(window.size(),1);
        std::vector<int> zeros = std::vector<int>(window.size(),0);
        SetValue(json, padding,             zeros)
        SetValue(json, stride,              ones)
        SetValue(json, upscale,             ones)
        SetValue(json, fwdAlgo,             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
        SetValue(json, bwdDataAlgo,         CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
        SetValue(json, bwdFilterAlgo,       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)

        init();
    };

    ConvolutionLayer(std::string name_,
                    int num_output_,
                    std::vector<int> window_,
                    std::vector<int> padding_, std::vector<int> stride_, std::vector<int> upscale_,
                    ComputeT weight_lr_mult_,   Filler weight_filler_, ComputeT weight_filler_param_,
                    ComputeT bias_lr_mult_,     Filler bias_filler_,   ComputeT  bias_filler_param_):
                    Layer(name_),
                    num_output(num_output_), window(window_), stride(stride_), padding(padding_), upscale(upscale_){

        weight_lr_mult = weight_lr_mult_;
        weight_filler = weight_filler_;
        weight_filler_param = weight_filler_param_;

        bias_lr_mult = bias_lr_mult_;
        bias_filler = bias_filler_;
        bias_filler_param = bias_filler_param_;

        init();
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        train_me = train_me && phase_ != Testing;

        std::cout<< (train_me? "* " : "  ");
        std::cout<<name;
        if (group>1) std::cout<<" ("<<group<<" groups)";

        if (in.size()==0) { std::cout<<std::endl<<"ConvolutionLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"ConvolutionLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        weight_dim[1] = in[0]->dim[1]/group;

        // create descriptor
        checkCUDNN(__LINE__,cudnnCreateFilterDescriptor(&filter_desc) );
        checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&bias_desc) );
        checkCUDNN(__LINE__,cudnnCreateConvolutionDescriptor(&conv_desc) );
        // set descriptor
        // set the parameters for convolution

        std::vector<int> weight_dim_group = weight_dim;
        weight_dim_group[0] = weight_dim[0]/group;

        checkCUDNN(__LINE__,cudnnSetFilterNdDescriptor(filter_desc,
                                                    CUDNNStorageT,
                                                    CUDNN_TENSOR_NCHW,
                                                    weight_dim.size(),
                                                    &weight_dim_group[0]) );

        checkCUDNN(__LINE__,cudnnSetConvolutionNdDescriptor(conv_desc,
                                                    padding.size(),
                                                    &padding[0],
                                                    &stride[0],
                                                    &upscale[0],
                                                    CUDNN_CROSS_CORRELATION,
                                                    CUDNNConvComputeT) );

        std::vector<int> bias_stride(bias_dim.size());

        bias_stride[bias_dim.size()-1] = 1;
        for (int d=bias_dim.size()-2;d>=0;--d){
            bias_stride[d] = bias_stride[d+1] *  bias_dim[d+1];
        }
        checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(bias_desc,
                                                    CUDNNStorageT,
                                                    bias_dim.size(),
                                                    &bias_dim[0],
                                                    &bias_stride[0]) );


        weight_numel = numel(weight_dim);
        bias_numel   = numel(bias_dim);

        if (weight_numel>0){
            std::cout<<" weight"; veciPrint(weight_dim);
            checkCUDA(__LINE__, cudaMalloc( &weight_dataGPU, weight_numel * sizeofStorageT) );
            memoryBytes += weight_numel * sizeofStorageT;
        }
        if (bias_numel>0){
            std::cout<<" bias"; veciPrint(bias_dim);
            checkCUDA(__LINE__, cudaMalloc( &bias_dataGPU, bias_numel * sizeofStorageT) );
            memoryBytes += bias_numel * sizeofStorageT;
        }
        std::cout<<std::endl;


        for (int i=0;i<out.size();++i){
            out[i]->need_diff = train_me || in[i]->need_diff; // if one of them need the grad

            std::vector<int> dimOut;
            dimOut.resize(in[i]->dim.size());

            checkCUDNN(__LINE__,cudnnGetConvolutionNdForwardOutputDim(conv_desc,
                                                                in[i]->getDesc(group),
                                                                filter_desc,
                                                                dimOut.size(),
                                                                &dimOut[0]
                                                                ));
            dimOut[1] *= group;

            size_t dall = in[i]->receptive_field.size();
            out[i]->receptive_field .resize(dall);
            out[i]->receptive_gap   .resize(dall);
            out[i]->receptive_offset.resize(dall);
            for(size_t d=0;d<dall;++d){
                out[i]->receptive_field[d] = in[i]->receptive_field[d] + ComputeT(window[d]-1) * in[i]->receptive_gap[d];
                out[i]->receptive_gap[d] = stride[d] * in[i]->receptive_gap[d];
                out[i]->receptive_offset[d] = in[i]->receptive_offset[d] - ComputeT(padding[d]) * in[i]->receptive_gap[d];
            }
            memoryBytes += out[i]->Malloc(dimOut);


        }

        // Allocate workspace
        fwdAlgoWorkspaces.resize(in.size());
        bwdDataAlgoWorkspaces.resize(out.size());
        bwdFilterAlgoWorkspaces.resize(out.size());

        fwdAlgoWorkspaceSizes.resize(in.size());
        bwdDataAlgoWorkspaceSizes.resize(out.size());
        bwdFilterAlgoWorkspaceSizes.resize(out.size());

        for (int i=0;i<in.size();++i){
            checkCUDNN(__LINE__,cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                                        in[i]->getDesc(group),
                                                                        filter_desc,
                                                                        conv_desc,
                                                                        out[i]->getDesc(group),
                                                                        fwdAlgo,
                                                                        &fwdAlgoWorkspaceSizes[i]));
            checkCUDA(__LINE__, cudaMalloc( &fwdAlgoWorkspaces[i], fwdAlgoWorkspaceSizes[i]) );
        }

        for (int i=0;i<out.size();++i){
            checkCUDNN(__LINE__,cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                                             filter_desc,
                                                                             out[i]->getDesc(group),
                                                                             conv_desc,
                                                                             in[i]->getDesc(group),
                                                                             bwdDataAlgo,
                                                                             &bwdDataAlgoWorkspaceSizes[i]));

            checkCUDNN(__LINE__,cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                                               in[i]->getDesc(group),
                                                                               out[i]->getDesc(group),
                                                                               conv_desc,
                                                                               filter_desc,
                                                                               bwdFilterAlgo,
                                                                               &bwdFilterAlgoWorkspaceSizes[i]));

            checkCUDA(__LINE__, cudaMalloc( &bwdDataAlgoWorkspaces[i], bwdDataAlgoWorkspaceSizes[i]) );
            checkCUDA(__LINE__, cudaMalloc( &bwdFilterAlgoWorkspaces[i], bwdFilterAlgoWorkspaceSizes[i]) );
        }

        return memoryBytes;
    };

    void forward(Phase phase_){

        for (int i=0;i<in.size();++i){
            for (int g = 0; g < group; g++) {
                checkCUDNN(__LINE__,cudnnConvolutionForward(cudnnHandle,
                                                      one,
                                                      in[i]->getDesc(group),
                                                      in[i]->dataGPU + (g * in[i]->sizeofitem() / group),
                                                      filter_desc,
                                                      weight_dataGPU + (g * weight_numel / group),
                                                      conv_desc,
                                                      fwdAlgo, // CUDNN For 3-d convolutions, only CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM is supported; support is provided for any format for srcDesc and destDesc as well as support for all data type configurations.
                                                      fwdAlgoWorkspaces[i],
                                                      fwdAlgoWorkspaceSizes[i],
                                                      zero,
                                                      out[i]->getDesc(group),
                                                      out[i]->dataGPU + (g * out[i]->sizeofitem() / group) ) );

            }

            if (bias_dim.size()<=5){ 
                checkCUDNN(__LINE__,cudnnAddTensor(cudnnHandle,
                                              one,
                                              bias_desc,
                                              bias_dataGPU,
                                              one,
                                              out[i]->desc,
                                              out[i]->dataGPU) );
            }else{
                std::vector<int> bias_dim_bug;
                bias_dim_bug.push_back(bias_dim[0]);
                bias_dim_bug.push_back(bias_dim[1]);
                bias_dim_bug.push_back(bias_dim[2]);
                bias_dim_bug.push_back(1);
                for (int d=3;d<bias_dim.size();++d) bias_dim_bug[3] *= bias_dim[d];
                std::vector<int> bias_stride(bias_dim_bug.size());
                bias_stride[bias_dim_bug.size()-1] = 1;
                for (int d=bias_dim_bug.size()-2;d>=0;--d){
                    bias_stride[d] = bias_stride[d+1] *  bias_dim_bug[d+1];
                }
                cudnnTensorDescriptor_t bias_desc_bug;
                checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&bias_desc_bug) );
                checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(bias_desc_bug,
                                                            CUDNNStorageT,
                                                            bias_dim_bug.size(),
                                                            &bias_dim_bug[0],
                                                            &bias_stride[0]) );
                std::vector<int> out_dim_bug;
                out_dim_bug.push_back(out[i]->dim[0]);
                out_dim_bug.push_back(out[i]->dim[1]);
                out_dim_bug.push_back(out[i]->dim[2]);
                out_dim_bug.push_back(1);
                for (int d=3;d<out[i]->dim.size();++d)  out_dim_bug[3] *= out[i]->dim[d];
                std::vector<int> strideA(out_dim_bug.size());
                strideA[out_dim_bug.size()-1] = 1;
                for (int d=out_dim_bug.size()-2;d>=0;--d)  strideA[d] = strideA[d+1] *  out_dim_bug[d+1];
                cudnnTensorDescriptor_t out_desc_bug;
                checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&out_desc_bug));
                checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(out_desc_bug,
                                                        CUDNNStorageT,
                                                        out_dim_bug.size(),
                                                        &out_dim_bug[0],
                                                        &strideA[0]) );
                checkCUDNN(__LINE__,cudnnAddTensor(cudnnHandle,
                                              one,
                                              bias_desc_bug,
                                              bias_dataGPU,
                                              one,
                                              out_desc_bug,
                                              out[i]->dataGPU) );
                checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(bias_desc_bug) );
                checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(out_desc_bug) );
            }
        }
    };
    void backward(Phase phase_){
        for (int i=0;i<out.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                for (int g = 0; g < group; g++) {
                    checkCUDNN(__LINE__,cudnnConvolutionBackwardData(cudnnHandle,
                                                              one,
                                                              filter_desc, weight_dataGPU + (g * weight_numel / group),
                                                              out[i]->getDesc(group), out[i]->diffGPU + (g * out[i]->sizeofitem() / group),
                                                              conv_desc,
                                                              bwdDataAlgo, bwdDataAlgoWorkspaces[i], bwdDataAlgoWorkspaceSizes[i],
                                                              one,
                                                              in[i]->getDesc(group), in[i]->diffGPU + (g * in[i]->sizeofitem() / group)));
                }
            }
        }
        // compute in->diff first because the next layer need to use it immediate, and because weight_diff needs to write to another GPU
        for (int i=0;i<out.size();++i){
            if (train_me){
                ComputeT beta = ComputeT(1);
                if (weight_numel>0){
                    for (int g = 0; g < group; g++) {
                        checkCUDNN(__LINE__,cudnnConvolutionBackwardFilter(cudnnHandle,
                                                                  one,
                                                                  in[i]->getDesc(group), in[i]->dataGPU + (g * in[i]->sizeofitem() / group),
                                                                  out[i]->getDesc(group), out[i]->diffGPU + (g * out[i]->sizeofitem() / group),
                                                                  conv_desc,
                                                                  bwdFilterAlgo, bwdFilterAlgoWorkspaces[i], bwdFilterAlgoWorkspaceSizes[i],
                                                                  &beta,
                                                                  filter_desc, weight_diffGPU + (g * weight_numel / group)));
                    }
                }
                if (bias_numel>0){
                    checkCUDNN(__LINE__,cudnnConvolutionBackwardBias(cudnnHandle,
                                                              one,
                                                              out[i]->desc,  out[i]->diffGPU,
                                                              &beta,
                                                              bias_desc, bias_diffGPU));
                }
            }
        }
    };
    ~ConvolutionLayer(){
        // destory the descriptor
        checkCUDNN(__LINE__,cudnnDestroyFilterDescriptor(filter_desc) );
        checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(bias_desc) );
        checkCUDNN(__LINE__,cudnnDestroyConvolutionDescriptor(conv_desc) );

        for (int i=0;i<in.size();++i){
            checkCUDA(__LINE__, cudaFree(fwdAlgoWorkspaces[i]));
        }
        for (int i=0;i<out.size();++i){
            checkCUDA(__LINE__, cudaFree(bwdDataAlgoWorkspaces[i]));
            checkCUDA(__LINE__, cudaFree(bwdFilterAlgoWorkspaces[i]));
        }
    };
};


class DeconvolutionLayer : public Layer {
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    std::vector<StorageT *> fwdAlgoWorkspaces;
    std::vector<StorageT *> bwdDataAlgoWorkspaces;
    std::vector<StorageT *> bwdFilterAlgoWorkspaces;

    std::vector<size_t> fwdAlgoWorkspaceSizes;
    std::vector<size_t> bwdDataAlgoWorkspaceSizes;
    std::vector<size_t> bwdFilterAlgoWorkspaceSizes;
public:
    cudnnConvolutionFwdAlgo_t fwdAlgo;
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;

    int num_output;
    std::vector<int> window;
    std::vector<int> stride;
    std::vector<int> padding;
    std::vector<int> upscale;
    int group;

    void init(){
        weight_dim.push_back(0);
        weight_dim.push_back(0);  // need the channel size from the input
        weight_dim.insert( weight_dim.end(), window.begin(), window.end() );

        bias_dim.resize(weight_dim.size(), 1);
        bias_dim[1] = num_output;
    };

    DeconvolutionLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, train_me,            true)
        SetOrDie(json, num_output           )
        SetOrDie(json, window               )
        SetValue(json, weight_lr_mult,      1.0)
        SetValue(json, weight_filler,       Xavier)
        SetValue(json, weight_filler_param, 0.0)
        SetValue(json, bias_lr_mult,        2.0)
        SetValue(json, bias_filler,         Constant)
        SetValue(json, bias_filler_param,   0.0)
        SetValue(json, weight_decay_mult,   1.0)
        SetValue(json, bias_decay_mult,     1.0)
        SetValue(json, group,               1)

        std::vector<int> ones  = std::vector<int>(window.size(),1);
        std::vector<int> zeros = std::vector<int>(window.size(),0);
        SetValue(json, padding,             zeros)
        SetValue(json, stride,              ones)
        SetValue(json, upscale,             ones)
        SetValue(json, fwdAlgo,             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
        SetValue(json, bwdDataAlgo,         CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
        SetValue(json, bwdFilterAlgo,       CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)

        init();
    };

    DeconvolutionLayer(std::string name_,
                    int num_output_,
                    std::vector<int> window_,
                    std::vector<int> padding_, std::vector<int> stride_, std::vector<int> upscale_,
                    ComputeT weight_lr_mult_,   Filler weight_filler_, ComputeT weight_filler_param_,
                    ComputeT bias_lr_mult_,     Filler bias_filler_,   ComputeT  bias_filler_param_):
                    Layer(name_),
                    num_output(num_output_), window(window_), stride(stride_), padding(padding_), upscale(upscale_){

        weight_lr_mult = weight_lr_mult_;
        weight_filler = weight_filler_;
        weight_filler_param = weight_filler_param_;

        bias_lr_mult = bias_lr_mult_;
        bias_filler = bias_filler_;
        bias_filler_param = bias_filler_param_;

        init();
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        train_me = train_me && phase_ != Testing;

        std::cout<< (train_me? "* " : "  ");
        std::cout<<name;
        if (group>1) std::cout<<" ("<<group<<" groups)";

        if (in.size()==0) { std::cout<<std::endl<<"DeconvolutionLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"DeconvolutionLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        weight_dim[0] = in[0]->dim[1];
        weight_dim[1] = num_output/group;

        // create descriptor
        checkCUDNN(__LINE__,cudnnCreateFilterDescriptor(&filter_desc) );
        checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&bias_desc) );
        checkCUDNN(__LINE__,cudnnCreateConvolutionDescriptor(&conv_desc) );
        // set descriptor
        // set the parameters for convolution

        std::vector<int> weight_dim_group = weight_dim;
        weight_dim_group[0] = weight_dim[0]/group;

        checkCUDNN(__LINE__,cudnnSetFilterNdDescriptor(filter_desc,
                                                    CUDNNStorageT,
                                                    CUDNN_TENSOR_NCHW,
                                                    weight_dim.size(),
                                                    &weight_dim_group[0]) );

        checkCUDNN(__LINE__,cudnnSetConvolutionNdDescriptor(conv_desc,
                                                    padding.size(),
                                                    &padding[0],
                                                    &stride[0],
                                                    &upscale[0],
                                                    CUDNN_CROSS_CORRELATION,
                                                    CUDNNConvComputeT) );

        std::vector<int> bias_stride(bias_dim.size());

        bias_stride[bias_dim.size()-1] = 1;
        for (int d=bias_dim.size()-2;d>=0;--d){
            bias_stride[d] = bias_stride[d+1] *  bias_dim[d+1];
        }
        checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(bias_desc,
                                                    CUDNNStorageT,
                                                    bias_dim.size(),
                                                    &bias_dim[0],
                                                    &bias_stride[0]) );

        weight_numel = numel(weight_dim);
        bias_numel   = numel(bias_dim);

        if (weight_numel>0){
            std::cout<<" weight"; veciPrint(weight_dim);
            checkCUDA(__LINE__, cudaMalloc( &weight_dataGPU, weight_numel * sizeofStorageT) );
            memoryBytes += weight_numel * sizeofStorageT;
        }
        if (bias_numel>0){
            std::cout<<" bias"; veciPrint(bias_dim);
            checkCUDA(__LINE__, cudaMalloc( &bias_dataGPU, bias_numel * sizeofStorageT) );
            memoryBytes += bias_numel * sizeofStorageT;
        }
        std::cout<<std::endl;


        for (int i=0;i<out.size();++i){
            out[i]->need_diff = train_me || in[i]->need_diff; // if one of them need the grad

            std::vector<int> dimOut;
            dimOut.resize(in[i]->dim.size());

            dimOut[0] = in[i]->dim[0];
            dimOut[1] = num_output;
            for (int d=0;d<window.size();++d){
                dimOut[2+d] = (in[i]->dim[2+d]-1)*stride[d] + window[d] - 2*padding[d];
            }

            size_t dall = in[i]->receptive_field.size();
            out[i]->receptive_field .resize(dall);
            out[i]->receptive_gap   .resize(dall);
            out[i]->receptive_offset.resize(dall);
            for(size_t d=0;d<dall;++d){
                out[i]->receptive_gap[d] = in[i]->receptive_gap[d] / stride[d];
                out[i]->receptive_field[d] = in[i]->receptive_field[d] - ComputeT(window[d]-1) * in[i]->receptive_gap[d];
                out[i]->receptive_offset[d] = in[i]->receptive_offset[d] + ComputeT(padding[d]) * in[i]->receptive_gap[d];
            }
            memoryBytes += out[i]->Malloc(dimOut);
        }

        // Allocate workspace
        fwdAlgoWorkspaces.resize(in.size());
        bwdDataAlgoWorkspaces.resize(out.size());
        bwdFilterAlgoWorkspaces.resize(out.size());

        fwdAlgoWorkspaceSizes.resize(in.size());
        bwdDataAlgoWorkspaceSizes.resize(out.size());
        bwdFilterAlgoWorkspaceSizes.resize(out.size());

        for (int i=0;i<in.size();++i){
            checkCUDNN(__LINE__,cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                                        out[i]->getDesc(group),
                                                                        filter_desc,
                                                                        conv_desc,
                                                                        in[i]->getDesc(group),
                                                                        fwdAlgo,
                                                                        &fwdAlgoWorkspaceSizes[i]));
            checkCUDA(__LINE__, cudaMalloc( &fwdAlgoWorkspaces[i], fwdAlgoWorkspaceSizes[i]) );
        }

        for (int i=0;i<out.size();++i){
            checkCUDNN(__LINE__,cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle,
                                                                             filter_desc,
                                                                             in[i]->getDesc(group),
                                                                             conv_desc,
                                                                             out[i]->getDesc(group),
                                                                             bwdDataAlgo,
                                                                             &bwdDataAlgoWorkspaceSizes[i]));

            checkCUDNN(__LINE__,cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle,
                                                                               out[i]->getDesc(group),
                                                                               in[i]->getDesc(group),
                                                                               conv_desc,
                                                                               filter_desc,
                                                                               bwdFilterAlgo,
                                                                               &bwdFilterAlgoWorkspaceSizes[i]));

            checkCUDA(__LINE__, cudaMalloc( &bwdDataAlgoWorkspaces[i], bwdDataAlgoWorkspaceSizes[i]) );
            checkCUDA(__LINE__, cudaMalloc( &bwdFilterAlgoWorkspaces[i], bwdFilterAlgoWorkspaceSizes[i]) );
        }

        return memoryBytes;
    };

    void forward(Phase phase_){

        for (int i=0;i<in.size();++i){
            for (int g = 0; g < group; g++) {
                checkCUDNN(__LINE__,cudnnConvolutionBackwardData(cudnnHandle,
                                                          one,
                                                          filter_desc, 
                                                          weight_dataGPU + (g * weight_numel / group),
                                                          in[i]->getDesc(group), 
                                                          in[i]->dataGPU + (g * in[i]->sizeofitem() / group),
                                                          conv_desc,
                                                          bwdDataAlgo, bwdDataAlgoWorkspaces[i], bwdDataAlgoWorkspaceSizes[i],
                                                          zero,
                                                          out[i]->getDesc(group),
                                                          out[i]->dataGPU + (g * out[i]->sizeofitem() / group)));
            }

            if (bias_dim.size()<=5){ 
                checkCUDNN(__LINE__,cudnnAddTensor(cudnnHandle,
                                              one,
                                              bias_desc,
                                              bias_dataGPU,
                                              one,
                                              out[i]->desc,
                                              out[i]->dataGPU) );
            }else{
                std::vector<int> bias_dim_bug;
                bias_dim_bug.push_back(bias_dim[0]);
                bias_dim_bug.push_back(bias_dim[1]);
                bias_dim_bug.push_back(bias_dim[2]);
                bias_dim_bug.push_back(1);
                for (int d=3;d<bias_dim.size();++d) bias_dim_bug[3] *= bias_dim[d];
                std::vector<int> bias_stride(bias_dim_bug.size());
                bias_stride[bias_dim_bug.size()-1] = 1;
                for (int d=bias_dim_bug.size()-2;d>=0;--d){
                    bias_stride[d] = bias_stride[d+1] *  bias_dim_bug[d+1];
                }
                cudnnTensorDescriptor_t bias_desc_bug;
                checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&bias_desc_bug) );
                checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(bias_desc_bug,
                                                            CUDNNStorageT,
                                                            bias_dim_bug.size(),
                                                            &bias_dim_bug[0],
                                                            &bias_stride[0]) );
                std::vector<int> out_dim_bug;
                out_dim_bug.push_back(out[i]->dim[0]);
                out_dim_bug.push_back(out[i]->dim[1]);
                out_dim_bug.push_back(out[i]->dim[2]);
                out_dim_bug.push_back(1);
                for (int d=3;d<out[i]->dim.size();++d)  out_dim_bug[3] *= out[i]->dim[d];
                std::vector<int> strideA(out_dim_bug.size());
                strideA[out_dim_bug.size()-1] = 1;
                for (int d=out_dim_bug.size()-2;d>=0;--d)  strideA[d] = strideA[d+1] *  out_dim_bug[d+1];
                cudnnTensorDescriptor_t out_desc_bug;
                checkCUDNN(__LINE__,cudnnCreateTensorDescriptor(&out_desc_bug));
                checkCUDNN(__LINE__,cudnnSetTensorNdDescriptor(out_desc_bug,
                                                        CUDNNStorageT,
                                                        out_dim_bug.size(),
                                                        &out_dim_bug[0],
                                                        &strideA[0]) );
                checkCUDNN(__LINE__,cudnnAddTensor(cudnnHandle,
                                              one,
                                              bias_desc_bug,
                                              bias_dataGPU,
                                              one,
                                              out_desc_bug,
                                              out[i]->dataGPU) );
                checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(bias_desc_bug) );
                checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(out_desc_bug) );
            }
        }
    };
    void backward(Phase phase_){
        for (int i=0;i<out.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                for (int g = 0; g < group; g++) {
                    checkCUDNN(__LINE__,cudnnConvolutionForward(cudnnHandle,
                                                          one,
                                                          out[i]->getDesc(group),
                                                          out[i]->diffGPU + (g * out[i]->sizeofitem() / group),
                                                          filter_desc,
                                                          weight_dataGPU + (g * weight_numel / group),
                                                          conv_desc,
                                                          fwdAlgo, // CUDNN For 3-d convolutions, only CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM is supported; support is provided for any format for srcDesc and destDesc as well as support for all data type configurations.
                                                          fwdAlgoWorkspaces[i],
                                                          fwdAlgoWorkspaceSizes[i],
                                                          one,
                                                          in[i]->getDesc(group),
                                                          in[i]->diffGPU + (g * in[i]->sizeofitem() / group)));
                }
            }
        }
        // compute in->diff first because the next layer need to use it immediate, and because weight_diff needs to write to another GPU
        for (int i=0;i<out.size();++i){
            if (train_me){
                ComputeT beta = ComputeT(1);
                if (weight_numel>0){
                    for (int g = 0; g < group; g++) {
                        checkCUDNN(__LINE__,cudnnConvolutionBackwardFilter(cudnnHandle,
                                                                  one,
                                                                  out[i]->getDesc(group), out[i]->diffGPU + (g * out[i]->sizeofitem() / group),
                                                                  in[i]->getDesc(group), in[i]->dataGPU + (g * in[i]->sizeofitem() / group),
                                                                  conv_desc,
                                                                  bwdFilterAlgo, bwdFilterAlgoWorkspaces[i], bwdFilterAlgoWorkspaceSizes[i],
                                                                  &beta,
                                                                  filter_desc, weight_diffGPU + (g * weight_numel / group)));
                    }
                }
                if (bias_numel>0){
                    checkCUDNN(__LINE__,cudnnConvolutionBackwardBias(cudnnHandle,
                                                              one,
                                                              out[i]->desc,  out[i]->diffGPU,
                                                              &beta,
                                                              bias_desc, bias_diffGPU));
                }
            }
        }
    };
    ~DeconvolutionLayer(){
        // destory the descriptor
        checkCUDNN(__LINE__,cudnnDestroyFilterDescriptor(filter_desc) );
        checkCUDNN(__LINE__,cudnnDestroyTensorDescriptor(bias_desc) );
        checkCUDNN(__LINE__,cudnnDestroyConvolutionDescriptor(conv_desc) );

        for (int i=0;i<in.size();++i){
            checkCUDA(__LINE__, cudaFree(fwdAlgoWorkspaces[i]));
        }
        for (int i=0;i<out.size();++i){
            checkCUDA(__LINE__, cudaFree(bwdDataAlgoWorkspaces[i]));
            checkCUDA(__LINE__, cudaFree(bwdFilterAlgoWorkspaces[i]));
        }
    };
};

class InnerProductLayer : public Layer {
    int num_input;
    int num_items;
public:
    int num_output;
    bool bias_term;

    StorageT* bias_multGPU; // a std::vector with size # of mini-batch training example

    InnerProductLayer(std::string name_,
                    int num_output_,
                    bool bias_term_=true,
                    ComputeT weight_lr_mult_=1.0,   Filler weight_filler_=Xavier, ComputeT weight_filler_param_=0.0,
                    ComputeT bias_lr_mult_=2.0,     Filler bias_filler_=Constant,   ComputeT  bias_filler_param_=0.0): Layer(name_),num_output(num_output_), bias_multGPU(NULL), bias_term(bias_term_){
        weight_filler = weight_filler_;
        weight_filler_param = weight_filler_param_;
        bias_filler = bias_filler_;
        bias_filler_param = bias_filler_param_;
        weight_lr_mult = weight_lr_mult_;
        bias_lr_mult   = bias_lr_mult_;
        train_me = true;
    };

    InnerProductLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, train_me,            true)
        SetValue(json, weight_lr_mult,      1.0)
        SetValue(json, weight_filler,       Xavier)
        SetValue(json, weight_filler_param, 0.0)
        SetValue(json, bias_lr_mult,        2.0)
        SetValue(json, bias_filler,         Constant)
        SetValue(json, bias_filler_param,   0.0)
        SetValue(json, weight_decay_mult,   1.0)
        SetValue(json, bias_decay_mult,     1.0)
        SetValue(json, bias_term,           true)
        SetOrDie(json, num_output           )

    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        train_me = train_me && phase_ != Testing;

        std::cout<< (train_me? "* " : "  ");
        std::cout<<name;

        if (in.size()==0) { std::cout<<std::endl<<"InnerProductLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"InnerProductLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        num_input = sizeofitem(in[0]->dim);
        num_items = in[0]->dim[0];

        weight_dim.resize(2);
        weight_dim[0] = num_output;
        weight_dim[1] = num_input;
        weight_numel = numel(weight_dim);

        if (bias_term){        
            bias_dim.resize(1);
            bias_dim[0] = num_output;
            bias_numel   = numel(bias_dim);
        }else{
            bias_numel   = 0;
        }


        if (weight_numel>0){
            std::cout<<" weight"; veciPrint(weight_dim);
            checkCUDA(__LINE__, cudaMalloc(&weight_dataGPU, weight_numel * sizeofStorageT) );
            memoryBytes += weight_numel * sizeofStorageT;
        }

        if (bias_numel>0){
            std::cout<<" bias"; veciPrint(bias_dim);
            checkCUDA(__LINE__, cudaMalloc(&bias_dataGPU, bias_numel * sizeofStorageT) );
            memoryBytes += bias_numel * sizeofStorageT;
            checkCUDA(__LINE__, cudaMalloc(&bias_multGPU, num_items * sizeofStorageT) );
            Kernel_set_value<<<CUDA_GET_BLOCKS(num_items), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(num_items), num_items, bias_multGPU, CPUCompute2StorageT(1));
            memoryBytes += num_items * sizeofStorageT;
        }
        std::cout<<std::endl;

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = train_me || in[i]->need_diff; // if one of them need the grad
            std::vector<int> dimOut(in[i]->dim.size());
            dimOut[0] = in[i]->dim[0];
            dimOut[1] = num_output;
            for (int d=2;d<in[i]->dim.size();++d)
                dimOut[d] = 1;

            size_t dall = in[i]->receptive_field.size();
            out[i]->receptive_field .resize(dall);
            out[i]->receptive_gap   .resize(dall);
            out[i]->receptive_offset.resize(dall);

            for(size_t d=0;d<dall;++d){
                out[i]->receptive_field[d] = in[i]->receptive_field[d] + ComputeT(in[i]->dim[d+2]-1) * in[i]->receptive_gap[d];
                out[i]->receptive_gap[d] = 0;
                out[i]->receptive_offset[d] = 0;

            }

            memoryBytes += out[i]->Malloc(dimOut);

        }
        return memoryBytes;
    };

    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
            checkCUBLAS(__LINE__, GPUgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, num_output, num_items, num_input, oneComputeT, weight_dataGPU, num_input, in[i]->dataGPU, num_input, zeroComputeT, out[i]->dataGPU, num_output) );
            if (bias_numel>0)
                checkCUBLAS(__LINE__, GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, num_output, num_items, 1, oneComputeT, bias_dataGPU, num_output, bias_multGPU, 1, oneComputeT, out[i]->dataGPU, num_output) );
        }
    };

    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            if (in[i]->need_diff){
                checkCUBLAS(__LINE__, GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, num_input, num_items, num_output, oneComputeT, weight_dataGPU, num_input, out[i]->diffGPU, num_output, oneComputeT, in[i]->diffGPU, num_input) );
            }
        }

        for (int i=0;i<in.size();++i){
            if (train_me){
                ComputeT beta = ComputeT(1);
                if (weight_numel>0){
                    checkCUBLAS(__LINE__, GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, num_input, num_output, num_items, oneComputeT, in[i]->dataGPU,  num_input, out[i]->diffGPU, num_output, &beta, weight_diffGPU, num_input) );
                }
                if (bias_numel>0){
                    checkCUBLAS(__LINE__, GPUgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, num_output,         1, num_items, oneComputeT, out[i]->diffGPU, num_output, bias_multGPU,    num_items, &beta, bias_diffGPU,    num_output) );
                }
            }
        }
    };

    ~InnerProductLayer(){
        if (bias_multGPU!=NULL) checkCUDA(__LINE__, cudaFree(bias_multGPU));
    };
};

class DropoutLayer: public Layer{
    std::vector<cudnnDropoutDescriptor_t> dropoutDescs;
    std::vector<StorageT *> states;
    std::vector<StorageT *> reserveSpaces;
    std::vector<size_t> stateSizes;
    std::vector<size_t> reserveSpaceSizes;

    std::vector<int > SIZEmask;
public:
    ComputeT dropout_rate;
    void init() {
        // This function is empty for now 
    };
    DropoutLayer(std::string name_, ComputeT dropout_rate_): Layer(name_), dropout_rate(dropout_rate_){
        init();
    };
    DropoutLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, dropout_rate,        0.5)
        init();
    };
    size_t Malloc(Phase phase_){
        dropoutDescs.resize(in.size());
        states.resize(in.size());
        reserveSpaces.resize(in.size());
        stateSizes.resize(in.size());
        reserveSpaceSizes.resize(in.size());
        
        for (int i=0;i<in.size();++i){
            checkCUDNN(__LINE__,cudnnCreateDropoutDescriptor(&dropoutDescs[i]));
        }
        
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;
        if (in.size()==0) { std::cout<<std::endl<<"DropoutLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"DropoutLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        SIZEmask.resize(out.size());
        for (int i=0;i<out.size();++i){
            SIZEmask [i] = numel(in[i]->dim);

            out[i]->need_diff = in[i]->need_diff;
            out[i]->receptive_field = in[i]->receptive_field;
            out[i]->receptive_gap = in[i]->receptive_gap;
            out[i]->receptive_offset = in[i]->receptive_offset;
            memoryBytes += out[i]->Malloc(in[i]->dim);
        }

        std::random_device rd;

        for (int i=0;i<in.size();++i){
            checkCUDNN(__LINE__,cudnnDropoutGetStatesSize(cudnnHandle, &stateSizes[i]));
            checkCUDA(__LINE__,cudaMalloc(&states[i], stateSizes[i]) );
            memoryBytes += stateSizes[i];
            checkCUDNN(__LINE__,cudnnDropoutGetReserveSpaceSize(in[i]->getDesc(), &reserveSpaceSizes[i]));
            checkCUDA(__LINE__,cudaMalloc(&reserveSpaces[i], reserveSpaceSizes[i]));
            memoryBytes += reserveSpaceSizes[i];
            checkCUDNN(__LINE__,cudnnSetDropoutDescriptor(dropoutDescs[i],
                                                          cudnnHandle,
                                                          dropout_rate,
                                                          states[i],
                                                          stateSizes[i],
                                                          rd()));
        }

        return memoryBytes;
    };
    ~DropoutLayer(){
        for (int i=0;i<in.size();++i){
            checkCUDNN(__LINE__,cudnnDestroyDropoutDescriptor(dropoutDescs[i]));
            checkCUDA(__LINE__, cudaFree(states[i]));
            checkCUDA(__LINE__, cudaFree(reserveSpaces[i]));
        }
    };
    void forward(Phase phase_){
        if ( phase_==Training ){
            for (int i=0;i<in.size();++i){
                checkCUDNN(__LINE__,cudnnDropoutForward(cudnnHandle,
                                                        dropoutDescs[i],
                                                        in[i]->getDesc(),
                                                        in[i]->dataGPU,
                                                        out[i]->getDesc(),
                                                        out[i]->dataGPU,
                                                        reserveSpaces[i],
                                                        reserveSpaceSizes[i]
                                                        ));
            }
        }else{
            for (int i=0;i<in.size();++i){
                if (out[i]!=in[i]){
                    checkCUDA(__LINE__,cudaMemcpy(out[i]->dataGPU, in[i]->dataGPU, sizeofStorageT*SIZEmask[i], cudaMemcpyDeviceToDevice));
                }
            }
        }
    };
    void backward(Phase phase_){
        if ( phase_==Training ){
            for (int i=0;i<in.size();++i){
                checkCUDNN(__LINE__,cudnnDropoutBackward(cudnnHandle,
                                                         dropoutDescs[i],
                                                         out[i]->getDesc(),
                                                         out[i]->diffGPU,
                                                         in[i]->getDesc(),
                                                         in[i]->diffGPU,
                                                         reserveSpaces[i],
                                                         reserveSpaceSizes[i]
                                                         ));
            }
        }else{
            std::cerr<<"there should be no backward for testing"<<std::endl;
            FatalError(__LINE__);
            for (int i=0;i<in.size();++i){
                if (out[i]!=in[i]){
                    checkCUDA(__LINE__,cudaMemcpy(in[i]->diffGPU, out[i]->diffGPU, sizeofStorageT*SIZEmask[i], cudaMemcpyDeviceToDevice));
                }
            }
        }
    };
};

class SoftmaxLayer : public Layer {
public:
    bool stable_gradient;

    SoftmaxLayer(std::string name_): Layer(name_), stable_gradient(true){};

    SoftmaxLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,           TrainingTesting)
        SetValue(json, stable_gradient, true)
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"SoftmaxLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"SoftmaxLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;
            out[i]->receptive_field = in[i]->receptive_field;
            out[i]->receptive_gap = in[i]->receptive_gap;
            out[i]->receptive_offset = in[i]->receptive_offset;
            memoryBytes += out[i]->Malloc(in[i]->dim);
        }
        return memoryBytes;
    };
    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
            checkCUDNN(__LINE__,cudnnSoftmaxForward(cudnnHandle,
                                              CUDNN_SOFTMAX_ACCURATE ,
                                              CUDNN_SOFTMAX_MODE_CHANNEL,
                                              one,
                                              in[i]->desc, in[i]->dataGPU,
                                              zero,
                                              out[i]->desc, out[i]->dataGPU));
        }
    };
    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                if (stable_gradient){
                    if (in[i]->diffGPU != out[i]->diffGPU){
                        xpy(numel(in[i]->dim), out[i]->diffGPU, in[i]->diffGPU);
                    }
                }else{
                    checkCUDNN(__LINE__,cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                                                      CUDNN_SOFTMAX_MODE_INSTANCE, //CUDNN_SOFTMAX_MODE_CHANNEL,
                                                      one,
                                                      out[i]->desc, out[i]->dataGPU, out[i]->desc, out[i]->diffGPU,
                                                      zero, //one, //bbb
                                                      in[i]->desc, in[i]->diffGPU));
                }
            }


        }
    };
};

class ActivationLayer : public Layer {
    cudnnActivationDescriptor_t activationDesc;
public:
    cudnnActivationMode_t mode;

    ActivationLayer(std::string name_, cudnnActivationMode_t mode_): Layer(name_), mode(mode_) {};

    ActivationLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, mode,                CUDNN_ACTIVATION_RELU)
        SetValue(json, phase,               TrainingTesting)
    };

    ~ActivationLayer() {
        checkCUDNN(__LINE__,cudnnDestroyActivationDescriptor(activationDesc));
    }

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        checkCUDNN(__LINE__,cudnnCreateActivationDescriptor(&activationDesc));
        checkCUDNN(__LINE__,cudnnSetActivationDescriptor(activationDesc, mode, CUDNN_PROPAGATE_NAN, 99.0)); // TODO: Refactor duplicate code

        if (in.size()==0) { std::cout<<std::endl<<"ActivationLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"ActivationLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;
            out[i]->receptive_field = in[i]->receptive_field;
            out[i]->receptive_gap = in[i]->receptive_gap;
            out[i]->receptive_offset = in[i]->receptive_offset;
            memoryBytes += out[i]->Malloc(in[i]->dim);
        }
        return memoryBytes;
    };
    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // CUDNN bug
            checkCUDNN(__LINE__,cudnnActivationForward(cudnnHandle,
                                                activationDesc,
                                                one,
                                                in[i]->desc, in[i]->dataGPU,
                                                zero,
                                                out[i]->desc, out[i]->dataGPU));
        }
    };
    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                checkCUDNN(__LINE__,cudnnActivationBackward(cudnnHandle,
                                                    activationDesc,
                                                    one,
                                                    out[i]->desc, out[i]->dataGPU, out[i]->desc, out[i]->diffGPU,
                                                    in[i]->desc, in[i]->dataGPU,
                                                    zero, //one, //bbb
                                                    in[i]->desc, in[i]->diffGPU));
            }
        }
    };
};

class PoolingLayer : public Layer {
    cudnnPoolingDescriptor_t desc;
public:
    cudnnPoolingMode_t mode;
    std::vector<int> window;
    std::vector<int> padding;
    std::vector<int> stride;

    void init(){
        checkCUDNN(__LINE__,cudnnCreatePoolingDescriptor(&desc) );
        checkCUDNN(__LINE__,cudnnSetPoolingNdDescriptor(desc,
                                                mode,
                                                CUDNN_PROPAGATE_NAN,
                                                window.size(),
                                                &window[0],
                                                &padding[0],
                                                &stride[0]));
    };

    PoolingLayer(std::string name_, cudnnPoolingMode_t mode_, std::vector<int> window_, std::vector<int> padding_, std::vector<int> stride_): Layer(name_), mode(mode_), window(window_), padding(padding_), stride(stride_){
        init();
    };

    PoolingLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, mode,                CUDNN_POOLING_MAX)
        SetOrDie(json, window               )
        std::vector<int> zeros = std::vector<int>(window.size(),0);
        SetValue(json, padding,             zeros)
        SetValue(json, stride,              window)

        init();
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes=0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"PoolingLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"PoolingLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;

            // compute the size to allocate memory
            std::vector<int> dimOut(in[i]->dim.size());
            dimOut[0] = in[i]->dim[0]; // size of mini-bath
            dimOut[1] = in[i]->dim[1]; // channels
            for (int d=2;d<in[i]->dim.size();++d){
              dimOut[d] = 1 + static_cast<int>(ceil(static_cast<float>(in[i]->dim[d] + 2*padding[d-2] - window[d-2])/stride[d-2]));
            }

            size_t dall = in[i]->receptive_field.size();
            out[i]->receptive_field .resize(dall);
            out[i]->receptive_gap   .resize(dall);
            out[i]->receptive_offset.resize(dall);
            for(size_t d=0;d<dall;++d){
                out[i]->receptive_field[d] = in[i]->receptive_field[d] + ComputeT(window[d]-1) * in[i]->receptive_gap[d];
                out[i]->receptive_gap[d] = stride[d] * in[i]->receptive_gap[d];
                out[i]->receptive_offset[d] = in[i]->receptive_offset[d] - ComputeT(padding[d]) * in[i]->receptive_gap[d];
            }

            memoryBytes += out[i]->Malloc(dimOut);
        }
        return memoryBytes;
    };
    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
            checkCUDNN(__LINE__,cudnnPoolingForward(cudnnHandle,
                                                desc,
                                                one,
                                                in[i]->desc, in[i]->dataGPU,
                                                zero,
                                                out[i]->desc, out[i]->dataGPU));

        }
    };
    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                checkCUDNN(__LINE__,cudnnPoolingBackward(cudnnHandle,
                                                    desc,
                                                    one,
                                                    out[i]->desc, out[i]->dataGPU, out[i]->desc, out[i]->diffGPU,
                                                    in[i]->desc, in[i]->dataGPU,
                                                    one, //zero, //one, //bbb
                                                    in[i]->desc, in[i]->diffGPU));
            }
        }
    };
    ~PoolingLayer(){
        checkCUDNN(__LINE__,cudnnDestroyPoolingDescriptor(desc) );
    };
};


class LRNLayer : public Layer {
    cudnnLRNDescriptor_t desc;
public:
    LRN mode;
    unsigned int local_size;
    ComputeT alpha;
    ComputeT beta;
    ComputeT k;

    void init(){
        if (local_size<CUDNN_LRN_MIN_N || local_size>CUDNN_LRN_MAX_N){ std::cout<<"LRN local_size out of range ["<< CUDNN_LRN_MIN_N <<","<< CUDNN_LRN_MAX_N <<"]: local_size="<<local_size<<std::endl; FatalError(__LINE__); }

        if (k<CUDNN_LRN_MIN_K){ std::cout<<"LRN k out of range ["<< CUDNN_LRN_MIN_K <<",Inf): k="<<k<<std::endl; FatalError(__LINE__); }

        if (beta<CUDNN_LRN_MIN_BETA){ std::cout<<"LRN beta out of range ["<< CUDNN_LRN_MIN_BETA <<",Inf): beta="<<beta<<std::endl; FatalError(__LINE__); }

        checkCUDNN(__LINE__,cudnnCreateLRNDescriptor(&desc) );
        checkCUDNN(__LINE__,cudnnSetLRNDescriptor(desc, local_size, (double)(alpha), (double)(beta), (double)(k)) );
    };

    ~LRNLayer(){
        checkCUDNN(__LINE__,cudnnDestroyLRNDescriptor(desc));
    };

    LRNLayer(std::string name_, LRN mode_, unsigned int local_size_, ComputeT alpha_, ComputeT beta_, ComputeT k_): Layer(name_), mode(mode_), local_size(local_size_), alpha(alpha_), beta(beta_), k(k_) {
        init();
    };

    LRNLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,           TrainingTesting)
        SetValue(json, mode,            CrossChannel)
        SetValue(json, local_size,      5)
        SetValue(json, alpha,           1e-4)
        SetValue(json, beta,            0.75)
        SetValue(json, k,               1.0)
        init();
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"LRNLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"LRNLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;
            out[i]->receptive_field = in[i]->receptive_field;
            out[i]->receptive_gap = in[i]->receptive_gap;
            out[i]->receptive_offset = in[i]->receptive_offset;
            memoryBytes += out[i]->Malloc(in[i]->dim);
        }
        return memoryBytes;
    };

    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
            switch(mode){
                case CrossChannel:
                    checkCUDNN(__LINE__,cudnnLRNCrossChannelForward(cudnnHandle, desc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                        one,
                                                        in[i]->desc, in[i]->dataGPU,
                                                        zero,
                                                        out[i]->desc, out[i]->dataGPU));
                break;
                case DivisiveNormalization:
#ifdef CUDNN_DivisiveNormalization
                // What is the Best Multi-Stage Architecture for Object Recognition?
                // http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
                    std::cout<<"Not implemented yet"<<std::endl;
                    FatalError(__LINE__);
                    checkCUDNN(__LINE__,cudnnDivisiveNormalizationForward(cudnnHandle, desc, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
                                                        one,
                                                        in[i]->desc, in[i]->dataGPU,
                                                        srcMeansData, tempData, tempData2,
                                                        zero,
                                                        out[i]->desc, out[i]->dataGPU));
#endif
                break;
            }
        }
    };

    void backward(Phase phase_){
        for (int i=0;i<in.size();++i){
            // if bottom still needs to compute gradients
            if (in[i]->need_diff){
                switch(mode){
                    case CrossChannel:
                        checkCUDNN(__LINE__,cudnnLRNCrossChannelBackward(cudnnHandle, desc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                                            one,
                                                            out[i]->desc /*srcDesc*/, out[i]->dataGPU /*srcData*/,
                                                            out[i]->desc /*srcDiffDesc*/, out[i]->diffGPU /*srcDiffData*/,
                                                            in[i]->desc /*destDesc*/, in[i]->dataGPU /*destData*/,
                                                            zero, //one, //bbb
                                                            in[i]->desc /*destDiffDesc*/, in[i]->diffGPU /*destDiffData*/));
                    break;
                    case DivisiveNormalization:
#ifdef CUDNN_DivisiveNormalization
                        std::cout<<"Not implemented yet"<<std::endl;
                        FatalError(__LINE__);
                        checkCUDNN(__LINE__,cudnnDivisiveNormalizationBackward(cudnnHandle, desc, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
                                                            one,
                                                            out[i]->desc /*srcDesc*/, out[i]->dataGPU /*srcData*/, srcMeansData /*srcMeansData*/,
                                                            out[i]->diffGPU /*srcDiffData*/,
                                                            tempData /*tempData*/, tempData2 /*tempData2*/,
                                                            zero, //one, //bbb
                                                            in[i]->desc /*destDataDesc*/, in[i]->diffGPU /*destDataDiff*/,
                                                            destMeansDiff /*destMeansDiff*/));
#endif
                    break;
                }
            }
        }
    };
};


class ReshapeLayer: public Layer {
public:
    std::vector<int> shape;
    ReshapeLayer(std::string name_, Phase phase_): Layer(name_){
        phase = phase_;
    };
    ReshapeLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, shape)
        bool remainExist = false;
        for(int d=0;d<shape.size();++d){
            if (shape[d]==-1){
                if (remainExist){
                    std::cerr<<"ReshapeLayer::shape can only have at most one -1"<<std::endl;
                    FatalError(__LINE__);
                }else{
                    remainExist = true;
                }
            }
        }
    };
    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"ReshapeLayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=out.size()) { std::cout<<std::endl<<"ReshapeLayer #in should be the same as #out"<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i]->need_diff;
            std::vector<int> dim;
            for(int d=0;d<shape.size();++d){
                if (shape[d]==0){
                    dim.push_back(in[i]->dim[d]);
                }else if (shape[d]==-1){
                    dim.push_back(-1);
                }else{
                    dim.push_back(shape[d]);
                }
            }
            int remain = numel(in[i]->dim)/numel(dim);
            if (remain!=1){
                remain = -remain;
                for(int d=0;d<dim.size();++d){
                    if (dim[d]==-1){
                        dim[d] = remain;
                    }
                }
            }

            out[i]->receptive_field = in[i]->receptive_field;
            out[i]->receptive_gap = in[i]->receptive_gap;
            out[i]->receptive_offset = in[i]->receptive_offset;
            memoryBytes += out[i]->Malloc(dim);
        }
        return memoryBytes;
    };

    void forward(Phase phase_){
        for (int i=0;i<in.size();++i){
            checkCUDA(__LINE__,cudaMemcpy(out[i]->dataGPU, in[i]->dataGPU, in[i]->numBytes(), cudaMemcpyDeviceToDevice));
        }
    };
    void backward(Phase phase_){
        for(int i=0;i<in.size();i++){
            if (in[i]->need_diff){
                size_t N = numel(in[i]->dim);
                Kernel_elementwise_acc<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, in[i]->diffGPU, out[i]->diffGPU);
            }
        }
    };
};

class ROILayer: public Layer {
public:
    std::vector<int> shape;
    ROILayer(std::string name_, Phase phase_): Layer(name_){
        phase = phase_;
    };
    ROILayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, shape)
    };
    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"ROILayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=2 * out.size()) { std::cout<<std::endl<<"ROILayer #in should be twice the size of #out"<<std::endl; FatalError(__LINE__); }
        if (in[0]->dim.size() != shape.size()+1) { std::cout<<std::endl<<"ROILayer's shape should be one dimension less than in, because the first dimension is the min-batch size."<<std::endl; FatalError(__LINE__); }

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i*2]->need_diff;

            std::vector<int> dim;
            dim.push_back(in[i*2]->dim[0]);

            for(int d=0;d<shape.size();++d){
                if (shape[d]==0){
                    dim.push_back(in[i*2]->dim[d+1]);
                }else{
                    dim.push_back(shape[d]);
                }
            }

            out[i]->receptive_field = in[i*2]->receptive_field;
            out[i]->receptive_gap = in[i*2]->receptive_gap;
            out[i]->receptive_offset = in[i*2]->receptive_offset;
            memoryBytes += out[i]->Malloc(dim);
        }
        return memoryBytes;
    };

    void forward(Phase phase_){
        for (int i=0;i<out.size();++i){
            size_t N = numel(out[i]->dim);
            switch(shape.size()){
                case 3:
                    Kernel_ROIforward_2D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, out[i]->dataGPU, in[i*2]->dataGPU, in[i*2+1]->dataGPU, out[i]->dim[1], out[i]->dim[2], out[i]->dim[3], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3]);
                break;
                case 4:
                    Kernel_ROIforward_3D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, out[i]->dataGPU, in[i*2]->dataGPU, in[i*2+1]->dataGPU, out[i]->dim[1], out[i]->dim[2], out[i]->dim[3], out[i]->dim[4], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], in[i*2]->dim[4]);
                break;
                case 5:
                    Kernel_ROIforward_4D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, out[i]->dataGPU, in[i*2]->dataGPU, in[i*2+1]->dataGPU, out[i]->dim[1], out[i]->dim[2], out[i]->dim[3], out[i]->dim[4], out[i]->dim[5], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], in[i*2]->dim[4], in[i*2]->dim[5]);
                break;
                default:
                    std::cerr<<"Haven't implemented yet"<<std::endl; FatalError(__LINE__);
                break;
            }
        }
    };
    void backward(Phase phase_){
        for(int i=0;i<out.size();i++){
            if (in[i*2]->need_diff){
                size_t N = numel(out[i]->dim);
                switch(shape.size()){
                    case 3:
                        Kernel_ROIbackward_2D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, out[i]->diffGPU, in[i*2]->diffGPU, in[i*2+1]->dataGPU, out[i]->dim[1], out[i]->dim[2], out[i]->dim[3], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3]);
                    break;
                    case 4:
                        Kernel_ROIbackward_3D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, out[i]->diffGPU, in[i*2]->diffGPU, in[i*2+1]->dataGPU, out[i]->dim[1], out[i]->dim[2], out[i]->dim[3], out[i]->dim[4], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], in[i*2]->dim[4]);
                    break;
                    case 5:
                        Kernel_ROIbackward_4D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, out[i]->diffGPU, in[i*2]->diffGPU, in[i*2+1]->dataGPU, out[i]->dim[1], out[i]->dim[2], out[i]->dim[3], out[i]->dim[4], out[i]->dim[5], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], in[i*2]->dim[4], in[i*2]->dim[5]);
                    break;
                    default:
                        std::cerr<<"Haven't implemented yet"<<std::endl; FatalError(__LINE__);
                    break;
                }
            }
        }
    };
};


class ROIPoolingLayer: public Layer {
    std::vector< size_t* > GPUIndex;
public:
    ComputeT spatial_scale;
    std::vector<int> shape;
    ROIPoolingLayer(std::string name_, Phase phase_): Layer(name_){
        phase = phase_;
    };
    ROIPoolingLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, shape)
        SetOrDie(json, spatial_scale)
    };
    ~ROIPoolingLayer(){
        for (int i=0;i<GPUIndex.size();++i){
            if (GPUIndex[i]!=NULL){
                checkCUDA(__LINE__, cudaFree(GPUIndex[i]) );
                GPUIndex[i] = NULL;
            }
        }
    };
    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        if (in.size()==0) { std::cout<<std::endl<<"ROILayer in shouldn't be empty"<<std::endl; FatalError(__LINE__); }
        if (in.size()!=2 * out.size()) { std::cout<<std::endl<<"ROILayer #in should be twice the size of #out"<<std::endl; FatalError(__LINE__); }
        if (in[0]->dim.size() != shape.size()+2) { std::cout<<std::endl<<"ROILayer's shape should be two dimensions less than in."<<std::endl; FatalError(__LINE__); }

        GPUIndex.resize(out.size(), NULL);

        for (int i=0;i<out.size();++i){
            out[i]->need_diff = in[i*2]->need_diff;

            if ( sizeofitem(in[i*2+1]->dim) != 1 + 2 * shape.size() ){
                std::cout<<std::endl<<"ROILayer in["<<i*2+1<<"]->dim is wrong"<<std::endl; FatalError(__LINE__);
            }

            std::vector<int> dim;
            dim.push_back(in[i*2+1]->dim[0]);   // number of boxes
            dim.push_back(in[i*2]->dim[1]);     // number of channels from convolutions
            for(int d=0;d<shape.size();++d){
                dim.push_back(shape[d]);
            }
            memoryBytes += out[i]->Malloc(dim);

            if (in[i*2]->need_diff){
                checkCUDA(__LINE__, cudaMalloc(&GPUIndex[i], numel(out[i]->dim) * sizeof(size_t)) );
                memoryBytes += numel(out[i]->dim) * sizeof(size_t);
            }
        }
        return memoryBytes;
    };

    void forward(Phase phase_){
        for (int i=0;i<out.size();++i){
            size_t N = numel(out[i]->dim);
            switch(shape.size()){
                case 2:
                    Kernel_ROIPoolForward_2D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, in[i*2]->dataGPU, in[i*2+1]->dataGPU, out[i]->dataGPU, GPUIndex[i], spatial_scale, in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], shape[0], shape[1]);
                break;
                case 3:
                    Kernel_ROIPoolForward_3D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, in[i*2]->dataGPU, in[i*2+1]->dataGPU, out[i]->dataGPU, GPUIndex[i], spatial_scale, in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], in[i*2]->dim[4], shape[0], shape[1], shape[2]);
                break;
                default:
                    std::cerr<<"Haven't implemented yet"<<std::endl; FatalError(__LINE__);
                break;
            }
        }
    };
    void backward(Phase phase_){
        for(int i=0;i<out.size();i++){
            if (in[i*2]->need_diff){
                size_t N = numel(in[i*2]->dim);
                switch(shape.size()){
                    case 2:
                        Kernel_ROIPoolBackward_2D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, in[i*2]->diffGPU, in[i*2+1]->dataGPU, out[i]->diffGPU, GPUIndex[i], spatial_scale, in[i*2+1]->dim[0], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], shape[0], shape[1]);
                    break;
                    case 3:
                        Kernel_ROIPoolBackward_3D<<<CUDA_GET_BLOCKS(N), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, in[i*2]->diffGPU, in[i*2+1]->dataGPU, out[i]->diffGPU, GPUIndex[i], spatial_scale, in[i*2+1]->dim[0], in[i*2]->dim[1], in[i*2]->dim[2], in[i*2]->dim[3], in[i*2]->dim[4], shape[0], shape[1], shape[2]);
                    break;
                    default:
                        std::cerr<<"Haven't implemented yet"<<std::endl; FatalError(__LINE__);
                    break;
                }
            }
        }
    };
};


class ElementWiseLayer : public Layer {
    int in_group;
public:
    ElementWiseOp mode;
    bool last_in_is_coeff;
    std::vector<ComputeT> coeff;

    ElementWiseLayer(std::string name_, ElementWiseOp mode_, bool last_in_is_coeff_=false): Layer(name_), mode(mode_), last_in_is_coeff(last_in_is_coeff_){
    };
    ElementWiseLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, mode)
        SetValue(json, last_in_is_coeff, false)
        SetValue(json, coeff,       std::vector<ComputeT>())
    };
    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        in_group = in.size()/out.size();
        if (in.size()!= in_group * out.size()){ std::cout<<"ElementWiseLayer in out size wrong "<<std::endl; FatalError(__LINE__); }

        if (coeff.size()==0) coeff = std::vector<ComputeT>(last_in_is_coeff? in_group-1: in_group,1);

        for(int j=0;j<out.size();j++){
            out[j]->need_diff = false;
            for(int i=j*in_group; i<(j+1)*in_group;i++){
                if (in[i]->need_diff){
                    out[j]->need_diff = true;
                    break;
                }
            }

            out[j]->receptive_field = in[j*in_group]->receptive_field;
            out[j]->receptive_gap = in[j*in_group]->receptive_gap;
            out[j]->receptive_offset = in[j*in_group]->receptive_offset;
            for(int i=j*in_group+1; i<(j+1)*in_group;i++){
                for(size_t d=0; d<out[j]->receptive_field.size();++d){
                    out[j]->receptive_field  [d] = max(out[j]->receptive_field  [d],in[i]->receptive_field  [d]);
                    out[j]->receptive_gap    [d] = max(out[j]->receptive_gap    [d],in[i]->receptive_gap    [d]);
                    out[j]->receptive_offset [d] = max(out[j]->receptive_offset [d],in[i]->receptive_offset [d]);
                }
            }

            memoryBytes += out[j]->Malloc(in[j*in_group]->dim);
        }
        return memoryBytes;
    };
    void forward(Phase phase_){
        switch(mode){
            case ElementWise_EQL:
                for (int i=0;i<out.size();++i){
                    int n = numel(out[i]->dim);
                    GPU_set_ones(n, out[i]->dataGPU);
                    for (int j=i*in_group+1; j<(i+1)*in_group; ++j){
                        GPU_elementwise_comparison(n, out[i]->dataGPU, in[i*in_group]->dataGPU, in[j]->dataGPU);
                    }
                }
            break;
            case ElementWise_MUL: std::cout<<"Not implemented yet"<<std::endl; FatalError(__LINE__);
            break;
            case ElementWise_SUM:
                for (int j=0;j<out.size();++j){
                    int i=j*in_group;
                    StorageT* coeff_data = last_in_is_coeff ? in[i + in_group - 1]->dataGPU : NULL;
                    size_t N = numel(in[i]->dim);
                    size_t items = in[i]->dim[0];
                    size_t dim = in[i]->sizeofitem();
                    CoeffElementWiseSumReplace<<<CUDA_GET_BLOCKS(N),CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, coeff[0], coeff_data, 0 * items, dim, in[i]->dataGPU, out[j]->dataGPU);
                    for (i=i+1; i<(j+1)*in_group - last_in_is_coeff; ++i){
                        size_t ii = i-j*in_group;
                        CoeffElementWiseSumAccumulate<<<CUDA_GET_BLOCKS(N),CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, coeff[ii], coeff_data, ii * items, dim, in[i]->dataGPU, out[j]->dataGPU);
                    }
                }
            break;
            case ElementWise_MIN: std::cout<<"Not implemented yet"<<std::endl; FatalError(__LINE__);
            break;
            case ElementWise_MAX: std::cout<<"Not implemented yet"<<std::endl; FatalError(__LINE__);
            break;
        };
    };
    void backward(Phase phase_){
        switch(mode){
            case ElementWise_EQL: std::cout<<"ElementWise_EQL cannot backprop"<<std::endl; FatalError(__LINE__);
            break;
            case ElementWise_MUL: std::cout<<"Not implemented yet"<<std::endl; FatalError(__LINE__);
            break;
            case ElementWise_SUM:
                for (int j=0;j<out.size();++j){
                    int i=j*in_group;
                    StorageT* coeff_data = last_in_is_coeff ? in[i + in_group - 1]->dataGPU : NULL;
                    size_t N = numel(in[i]->dim);
                    size_t items = in[i]->dim[0];
                    size_t dim = in[i]->sizeofitem();
                    for (; i<(j+1)*in_group - last_in_is_coeff; ++i){
                        size_t ii = i-j*in_group;
                        CoeffElementWiseSumAccumulate<<<CUDA_GET_BLOCKS(N),CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(N), N, coeff[ii], coeff_data, ii * items, dim, out[j]->diffGPU, in[i]->diffGPU);
                    }
                }
            break;
            case ElementWise_MIN: std::cout<<"Not implemented yet"<<std::endl; FatalError(__LINE__);
            break;
            case ElementWise_MAX: std::cout<<"Not implemented yet"<<std::endl; FatalError(__LINE__);
            break;
        };
    };
};


class ConcatLayer: public Layer {
    int in_group;
public:
    ConcatLayer(std::string name_, Phase phase_): Layer(name_){
        phase = phase_;
    };
    ConcatLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
    };
    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        std::cout<< (train_me? "* " : "  ");
        std::cout<<name<<std::endl;

        in_group = in.size()/out.size();
        if (in_group<2 || in.size()!= in_group * out.size()){ std::cout<<"ElementWiseLayer in out size wrong "<<std::endl; FatalError(__LINE__); }

        for(int j=0;j<out.size();j++){
            out[j]->need_diff = false;
            for(int i=j*in_group; i<(j+1)*in_group;i++){
                if (in[i]->need_diff){
                    out[j]->need_diff = true;
                    break;
                }
            }
            std::vector<int> dim = in[j*in_group]->dim;
            for(int i=j*in_group+1; i<(j+1)*in_group;i++){
                dim[1] += in[i]->dim[1];
            }

            out[j]->receptive_field = in[j*in_group]->receptive_field;
            out[j]->receptive_gap = in[j*in_group]->receptive_gap;
            out[j]->receptive_offset = in[j*in_group]->receptive_offset;
            for(int i=j*in_group+1; i<(j+1)*in_group;i++){
                for(size_t d=0; d<out[j]->receptive_field.size();++d){
                    out[j]->receptive_field[d]  = max(out[j]->receptive_field [d],in[i]->receptive_field [d]);
                    out[j]->receptive_gap  [d]  = max(out[j]->receptive_gap   [d],in[i]->receptive_gap   [d]);
                    out[j]->receptive_offset[d] = min(out[j]->receptive_offset[d],in[i]->receptive_offset[d]);
                }
            }

            memoryBytes += out[j]->Malloc(dim);
        }
        return memoryBytes;
    };
    void forward(Phase phase_){
        for(int j=0;j<out.size();j++){
            int offset = 0;
            int numofitems = out[j]->dim[0];
            for(int i=j*in_group; i<(j+1)*in_group;i++){
                copyGPUforward (numofitems, in[i]->dataGPU, out[j]->dataGPU, sizeofitem(in[i]->dim), sizeofitem(out[j]->dim), offset);
                offset += sizeofitem(in[i]->dim);
            }
        }
    };
    void backward(Phase phase_){
        for(int j=0;j<out.size();j++){
            int offset = 0;
            int numofitems = out[j]->dim[0];
            for(int i=j*in_group; i<(j+1)*in_group;i++){
                if (in[i]->need_diff){
                    copyGPUbackward(numofitems, in[i]->diffGPU, out[j]->diffGPU, sizeofitem(in[i]->dim), sizeofitem(out[j]->dim), offset);
                }
                offset += sizeofitem(in[i]->dim);
            }
        }
    };
};


class LossLayer : public Layer {
    StorageT* loss_values;
    StorageT* loss_weightsGPU;
    size_t loss_numel;
    int numExamples;
    ComputeT scale;
public:
    ComputeT result;
    ComputeT loss;


    LossObjective mode;
    ComputeT loss_weight;
    std::vector<ComputeT> loss_weights;
    ComputeT margin;

    LossLayer(std::string name_, LossObjective mode_, ComputeT loss_weight_)
        : Layer(name_), mode(mode_), loss_weight(loss_weight_),
          loss_values(NULL), loss_weightsGPU(NULL) {
        train_me = false;
    };

    LossLayer(JSON* json): loss_values(NULL), loss_weightsGPU(NULL){
        SetOrDie(json, name)
        SetValue(json, phase,       TrainingTesting)
        SetOrDie(json, mode)
        SetValue(json, loss_weight, 1)
        SetValue(json, margin,      1)

        SetValue(json, loss_weights,  std::vector<ComputeT>())
        train_me = false;
    };

    ~LossLayer() {
        if (loss_values != NULL)
            checkCUDA(__LINE__, cudaFree(loss_values));
        if (loss_weightsGPU != NULL)
            checkCUDA(__LINE__, cudaFree(loss_weightsGPU));
    };

    size_t Malloc(Phase phase_) {
        std::cout << (train_me ? "* " : "  ");
        std::cout << name << std::endl;

        size_t memoryBytes = 0;

        numExamples = in[0]->dim[0];

        switch (mode) {
            case MultinomialLogistic_StableSoftmax:
            case MultinomialLogistic:
                if (!(in.size() == 2 || in.size() == 3)) {
                    std::cout <<
                    "LossLayer: MultinomialLogistic should have 2 or 3 ins" <<
                    std::endl;
                    FatalError(__LINE__);
                }
            if (!same_dim_EC(in[0]->dim, in[1]->dim)) {
                std::cout <<
                "LossLayer: MultinomialLogistic should have the same dimensions except channels" <<
                std::endl;
                FatalError(__LINE__);
            }
            if (in[1]->dim[1] != 1) {
                std::cout <<
                "LossLayer: MultinomialLogistic in[1] should have only 1 channel" <<
                std::endl;
                FatalError(__LINE__);
            }
            if (in.size() == 3 && !(numel(in[0]->dim) == numel(in[2]->dim) ||
                                    sizeofitem(in[0]->dim) ==
                                    numel(in[2]->dim))) {
                std::cout <<
                "LossLayer: MultinomialLogistic in[2] size should be either the same with in[0] or should be the same with sizeofitem for in[0]" <<
                std::endl;
                FatalError(__LINE__);
            }
            loss_numel = numExamples * numspel(in[0]->dim);
            break;
            case SmoothL1:
                if (!(in.size() == 2 || in.size() == 3)) {
                    std::cout << "LossLayer: SmoothL1 should have 2 or 3 ins" <<
                    std::endl;
                    FatalError(__LINE__);
                }
                if (!same_dim(in[0]->dim, in[1]->dim)) {
                    std::cout <<
                    "LossLayer: SmoothL1 should have the same dimensions" <<
                    std::endl;
                    FatalError(__LINE__);
                }
                if (in.size() == 3 && !same_dim(in[0]->dim, in[2]->dim)) {
                    std::cout <<
                    "LossLayer: SmoothL1 should have the same dimensions" <<
                    std::endl;
                    FatalError(__LINE__);
                }
                loss_numel = numel(in[0]->dim);
            break;
            case Contrastive:
                loss_numel = numExamples;
            break;
            case EuclideanSSE:
                if (!(in.size() == 2 || in.size() == 3)) {
                    std::cout << "LossLayer: EuclideanSSE should have 2 or 3 ins" <<
                    std::endl;
                    FatalError(__LINE__);
                }
                if (!same_dim(in[0]->dim, in[1]->dim)) {
                    std::cout <<
                    "LossLayer: EuclideanSSE should have the same dimensions" <<
                    std::endl;
                    FatalError(__LINE__);
                }
                if (in.size() == 3 && !same_dim(in[0]->dim, in[2]->dim)) {
                    std::cout <<
                    "LossLayer: EuclideanSSE should have the same dimensions" <<
                    std::endl;
                    FatalError(__LINE__);
                }
                loss_numel = numel(in[0]->dim);
                break;
            case HingeL1:
                break;
            case HingeL2:
                break;
            case SigmoidCrossEntropy:
                break;
            case Infogain:
                break;
        }
        scale = loss_weight / loss_numel;

        memoryBytes += loss_numel * sizeofStorageT;
        checkCUDA(__LINE__, cudaMalloc(&loss_values, memoryBytes));


        if (loss_weights.size() > 0) {
            size_t newBytes = loss_weights.size() * sizeofStorageT;
            checkCUDA(__LINE__, cudaMalloc(&loss_weightsGPU, newBytes));
            memoryBytes += newBytes;

            StorageT *CPUram = new StorageT[loss_weights.size()];
            for (int i = 0; i < loss_weights.size(); ++i) {
                CPUram[i] = CPUCompute2StorageT(loss_weights[i]);
            }
            checkCUDA(__LINE__, cudaMemcpy(loss_weightsGPU, CPUram, newBytes,
                                           cudaMemcpyHostToDevice));
            delete[] CPUram;
        }

        return memoryBytes;
    };

    void display() {
        std::cout << " loss = " << loss;
        std::cout << " * " << loss_weight;
        if (mode == MultinomialLogistic_StableSoftmax ||
            mode == MultinomialLogistic)
            std::cout << "  eval = " << result;
        std::cout << "   ";
    };

    void eval(){
        ComputeT resultSum;
        switch(mode){
            case MultinomialLogistic_StableSoftmax:
            case MultinomialLogistic:
                Accuracy_MultinomialLogistic<<<CUDA_GET_BLOCKS(loss_numel),
                    CUDA_NUM_THREADS>>>(
                        CUDA_GET_LOOPS(loss_numel),
                        loss_numel, in[0]->dim[1], numspel(in[0]->dim),
                        (in.size()==3 ? numel(in[2]->dim) : 0),
                        in[0]->dataGPU, in[1]->dataGPU, loss_weightsGPU,
                        (in.size()==3 ? in[2]->dataGPU : NULL),
                        loss_values);
                checkCUBLAS(__LINE__, GPUasum(cublasHandle, loss_numel,
                                              loss_values, 1, &resultSum));
                result += resultSum / loss_numel;
                Loss_MultinomialLogistic<<<CUDA_GET_BLOCKS(loss_numel),
                    CUDA_NUM_THREADS>>>(
                        CUDA_GET_LOOPS(loss_numel),
                        loss_numel, in[0]->dim[1], numspel(in[0]->dim),
                        (in.size()==3 ? numel(in[2]->dim) : 0),
                        in[0]->dataGPU, in[1]->dataGPU, loss_weightsGPU,
                        (in.size()==3 ? in[2]->dataGPU : NULL),
                        loss_values);
                break;
            case SmoothL1:
                Loss_SmoothL1<<<CUDA_GET_BLOCKS(loss_numel),
                    CUDA_NUM_THREADS>>>(
                        CUDA_GET_LOOPS(loss_numel),
                        loss_numel, in[0]->dataGPU, in[1]->dataGPU,
                        (in.size()==3 ? in[2]->dataGPU : NULL), loss_values);
                break;
            case Contrastive:
                Loss_Contrastive<<<CUDA_GET_BLOCKS(numExamples),
                    CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(loss_numel),
                        loss_numel, in[0]->dim[1],
                        margin,in[0]->dataGPU, in[1]->dataGPU, in[2]->dataGPU,
                        loss_values);
                break;
            case EuclideanSSE:
                Loss_EuclideanSSE<<<CUDA_GET_BLOCKS(loss_numel),
                    CUDA_NUM_THREADS>>>(
                        CUDA_GET_LOOPS(loss_numel),
                        loss_numel, in[0]->dataGPU, in[1]->dataGPU,
                        (in.size()==3 ? in[2]->dataGPU : NULL), loss_values);
                break;
            case HingeL1:
                break;
            case HingeL2:
                break;
            case SigmoidCrossEntropy:
                break;
            case Infogain:
                break;
        }
        ComputeT lossSum;
        checkCUBLAS(__LINE__, GPUasum(cublasHandle, loss_numel,
                                      loss_values, 1, &lossSum));
        loss += lossSum/loss_numel;
    };


    void backward(Phase phase_){
        // either write this in Cuda or get both the prediction and ground truth to CPU and do the computation and write the diff back to GPU
        if (in[0]->need_diff){
            switch(mode){
                case MultinomialLogistic_StableSoftmax:
                    LossGrad_MultinomialLogistic_StableSoftmax<<<
                        CUDA_GET_BLOCKS(loss_numel), CUDA_NUM_THREADS>>>(
                            CUDA_GET_LOOPS(loss_numel),
                            loss_numel, in[0]->dim[1], numspel(in[0]->dim),
                            (in.size()==3 ? numel(in[2]->dim) : 0), scale,
                            in[0]->dataGPU, in[1]->dataGPU, loss_weightsGPU,
                            (in.size()==3 ? in[2]->dataGPU : NULL),
                            in[0]->diffGPU);
                    break;
                case MultinomialLogistic:
                    LossGrad_MultinomialLogistic<<<
                        CUDA_GET_BLOCKS(loss_numel), CUDA_NUM_THREADS>>>(
                            CUDA_GET_LOOPS(loss_numel),
                            loss_numel, in[0]->dim[1], numspel(in[0]->dim),
                            (in.size()==3 ? numel(in[2]->dim) : 0), scale,
                            in[0]->dataGPU, in[1]->dataGPU, loss_weightsGPU,
                            (in.size()==3 ? in[2]->dataGPU : NULL),
                            in[0]->diffGPU);
                    break;
                case SmoothL1:
                    LossGrad_SmoothL1<<<CUDA_GET_BLOCKS(loss_numel),
                        CUDA_NUM_THREADS>>>(
                            CUDA_GET_LOOPS(loss_numel),
                            loss_numel, scale, in[0]->dataGPU, in[1]->dataGPU,
                            (in.size()==3 ? in[2]->dataGPU : NULL),
                            in[0]->diffGPU);
                    break;
                case Contrastive:
                    LossGrad_Contrastive<<<CUDA_GET_BLOCKS(numExamples),
                        CUDA_NUM_THREADS>>>(
                            CUDA_GET_LOOPS(loss_numel),
                            loss_numel, in[0]->dim[1], margin, scale,
                            in[0]->dataGPU, in[1]->dataGPU, in[2]->dataGPU,
                            in[0]->diffGPU, in[1]->diffGPU);
                    break;
                case EuclideanSSE:
                    LossGrad_EuclideanSSE<<<CUDA_GET_BLOCKS(loss_numel),
                        CUDA_NUM_THREADS>>>(
                            CUDA_GET_LOOPS(loss_numel),
                            loss_numel, scale, in[0]->dataGPU, in[1]->dataGPU,
                            (in.size()==3 ? in[2]->dataGPU : NULL),
                            in[0]->diffGPU);
                    break;
                case HingeL1:
                    break;
                case HingeL2:
                    break;
                case SigmoidCrossEntropy:
                    break;
                case Infogain:
                    break;
            }

        }
    };
};


/* ----------------------------------------------------------------------------
 * The following LSTM implementation are largely based on LRCN on Caffe.
 *
 * Project page: http://jeffdonahue.com/lrcn/
 * GitHub page:  https://github.com/LisaAnne/lisa-caffe-public
 * License page: https://github.com/BVLC/caffe/blob/master/LICENSE
 * ----------------------------------------------------------------------------
 */

__device__ ComputeT sigmoid(const ComputeT x) {
  return ComputeT(1) / (ComputeT(1) + exp(-x));
}

__device__ ComputeT tanh(const ComputeT x) {
  return ComputeT(2) * sigmoid(ComputeT(2) * x) - ComputeT(1);
}

__global__ void LSTMActsForward(size_t CUDA_NUM_LOOPS, size_t N, const int dim, const StorageT* X, StorageT* X_acts) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){
        const int x_dim = 4 * dim;
        const int d = index % x_dim;
        if (d < 3 * dim) {
            X_acts[index] = GPUCompute2StorageT(sigmoid(GPUStorage2ComputeT(X[index])));
        } else {
            X_acts[index] = GPUCompute2StorageT(tanh(GPUStorage2ComputeT(X[index])));
        }
    }
}

__global__ void LSTMUnitForward(size_t CUDA_NUM_LOOPS, size_t N, const size_t dim, const StorageT* C_prev, const StorageT* X, const StorageT* flush, StorageT* C, StorageT* H) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){
        const size_t n = index / dim;
        const size_t d = index % dim;
        const size_t offset = 4 * dim * n;
        const ComputeT i = GPUStorage2ComputeT(X[offset + d]);
        const ComputeT f = GPUStorage2ComputeT(X[offset + 1 * dim + d]);
        const ComputeT o = GPUStorage2ComputeT(X[offset + 2 * dim + d]);
        const ComputeT g = GPUStorage2ComputeT(X[offset + 3 * dim + d]);
        const ComputeT c = GPUStorage2ComputeT(flush[n]) * f * GPUStorage2ComputeT(C_prev[index]) + i * g;
        C[index] = GPUCompute2StorageT(c);
        H[index] = GPUCompute2StorageT(o * tanh(c));
    }
}

__global__ void LSTMUnitBackward(size_t CUDA_NUM_LOOPS, size_t N, const size_t dim, const StorageT* C_prev, const StorageT* X, const StorageT* C, const StorageT* H, const StorageT* flush, const StorageT* C_diff, const StorageT* H_diff, StorageT* C_prev_diff, StorageT* X_diff) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){
        const size_t n = index / dim;
        const size_t d = index % dim;
        const size_t offset = 4 * dim * n;
        const ComputeT i = GPUStorage2ComputeT(X[offset + d]);
        const ComputeT f = GPUStorage2ComputeT(X[offset + 1 * dim + d]);
        const ComputeT o = GPUStorage2ComputeT(X[offset + 2 * dim + d]);
        const ComputeT g = GPUStorage2ComputeT(X[offset + 3 * dim + d]);
        const ComputeT c_prev = GPUStorage2ComputeT(C_prev[index]);
        const ComputeT c = GPUStorage2ComputeT(C[index]);
        const ComputeT tanh_c = tanh(c);

        ComputeT h_diff = GPUStorage2ComputeT(H_diff[index]);
        const ComputeT c_term_diff = GPUStorage2ComputeT(C_diff[index]) + h_diff * o * (1 - tanh_c * tanh_c);
        const ComputeT flush_n = GPUStorage2ComputeT(flush[n]);

        C_prev_diff[index] = GPUCompute2StorageT(flush_n * c_term_diff * f);
        const size_t diff_offset = 4 * dim * n;
        X_diff[diff_offset + d] = GPUCompute2StorageT(c_term_diff * g);
        X_diff[diff_offset + 1 * dim + d] = GPUCompute2StorageT(flush_n * c_term_diff * c_prev);
        X_diff[diff_offset + 2 * dim + d] = GPUCompute2StorageT(h_diff * tanh_c);
        X_diff[diff_offset + 3 * dim + d] = GPUCompute2StorageT(c_term_diff * i);
    }
}

__global__ void LSTMActsBackward(size_t CUDA_NUM_LOOPS, size_t N, const size_t dim, const StorageT* X_acts, const StorageT* X_acts_diff, StorageT* X_diff) {
    const size_t idxBase = size_t(CUDA_NUM_LOOPS) * (size_t(CUDA_NUM_THREADS) * size_t(blockIdx.x) + size_t(threadIdx.x));
    if (idxBase >= N) return;
    for (size_t index = idxBase; index < min(N,idxBase+CUDA_NUM_LOOPS); ++index ){
        const size_t x_dim = 4 * dim;
        const size_t d = index % x_dim;
        const ComputeT X_act = GPUStorage2ComputeT(X_acts[index]);
        if (d < 3 * dim) {
            X_diff[index] = GPUCompute2StorageT(GPUStorage2ComputeT(X_acts_diff[index]) * X_act * (ComputeT(1) - X_act));
        } else {
            X_diff[index] = GPUCompute2StorageT(GPUStorage2ComputeT(X_acts_diff[index]) * (ComputeT(1) - X_act * X_act));
        }
    }
}

// A helper for LSTMLayer: computes a single timestep of the non-linearity of the LSTM, producing the updated cell and hidden states.
class LSTMUnitLayer : public Layer {
    size_t X_count;
    size_t count;
    size_t hidden_dim;
public:
    StorageT* X_acts;
    StorageT* X_acts_diff;
    LSTMUnitLayer(std::string name_): Layer(name_), X_acts(NULL), X_acts_diff(NULL){};
    size_t Malloc(Phase phase_) {
        std::cout << (train_me ? "* " : "  ");
        std::cout << name << std::endl;
        size_t memoryBytes = 0;
        const size_t X_bytes = in[1]->numBytes();
        checkCUDA(__LINE__, cudaMalloc(&X_acts_diff, X_bytes) );
        X_count = numel( in[1]->dim);
        count = numel(out[1]->dim);
        hidden_dim = numspel(in[0]->dim);
        return memoryBytes;
    };
    ~LSTMUnitLayer(){
        
        if (X_acts_diff!=NULL)  checkCUDA(__LINE__, cudaFree(X_acts_diff));
    };
    void forward(Phase phase_){
        LSTMActsForward<<<CUDA_GET_BLOCKS(X_count), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(X_count), X_count, hidden_dim, in[1]->dataGPU, X_acts);
        LSTMUnitForward<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(count), count, hidden_dim, in[0]->dataGPU, X_acts, in[2]->dataGPU, out[0]->dataGPU, out[1]->dataGPU);
    };
    void backward(Phase phase_){
        LSTMUnitBackward<<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(count), count, hidden_dim, in[0]->dataGPU, X_acts, out[0]->dataGPU, out[1]->dataGPU, in[2]->dataGPU, out[0]->diffGPU, out[1]->diffGPU, in[0]->diffGPU, X_acts_diff);
        LSTMActsBackward<<<CUDA_GET_BLOCKS(X_count), CUDA_NUM_THREADS>>>(CUDA_GET_LOOPS(X_count), X_count, hidden_dim, X_acts, X_acts_diff, in[1]->diffGPU);
    };
};

class LSTMLayer : public Layer {
    int batch_size_N;
    int seq_length_T;

    Response* pResponse_W_xc_x_static;
    Response* pResponse_W_xc_x;
    std::vector<Response*> responses_W_xc_x_;
    std::vector<Response*> responses_cont_;
    std::vector<Response*> responses_c_;
    std::vector<Response*> responses_h_;
    std::vector<Response*> responses_h_conted_;
    std::vector<Response*> responses_W_hc_h_;
    std::vector<Response*> responses_gate_input_;

    Response* in0;

    Layer* pLayer_x_transform;
    Layer* pLayer_x_static_transform;
    Layer* pLayer_h_conted;
    Layer* pLayer_transform;
    Layer* pLayer_gate_input;
    LSTMUnitLayer* pLayer_lstm_unit;

    std::vector<StorageT*> X_acts_;

    bool debug_mode;

public:
    int num_output;

    LSTMLayer(std::string name_,
            int num_output_,
            ComputeT weight_lr_mult_,   Filler weight_filler_, ComputeT weight_filler_param_,
            ComputeT bias_lr_mult_,     Filler bias_filler_,   ComputeT  bias_filler_param_): Layer(name_),num_output(num_output_),debug_mode(false){
        weight_filler = weight_filler_;
        weight_filler_param = weight_filler_param_;
        bias_filler = bias_filler_;
        bias_filler_param = bias_filler_param_;
        weight_lr_mult = weight_lr_mult_;
        bias_lr_mult   = bias_lr_mult_;
        train_me = true;
    };

    LSTMLayer(JSON* json){
        SetOrDie(json, name)
        SetValue(json, phase,               TrainingTesting)
        SetValue(json, train_me,            true)
        SetValue(json, weight_lr_mult,      1.0)
        SetValue(json, weight_filler,       Xavier)
        SetValue(json, weight_filler_param, 0.0)
        SetValue(json, bias_lr_mult,        2.0)
        SetValue(json, bias_filler,         Constant)
        SetValue(json, bias_filler_param,   0.0)
        SetValue(json, weight_decay_mult,   1.0)
        SetValue(json, bias_decay_mult,     1.0)
        SetOrDie(json, num_output           )
        SetValue(json, debug_mode,          false)
    };

    size_t FillUnrolledNet(Phase phase_){

        size_t memoryBytes = 0;
        std::vector<int> dim;

        // first h_0 is not part of the output
        dim.clear();
        dim.push_back(1);
        dim.push_back(batch_size_N);
        dim.push_back(num_output);

        Response* pResponse_h_0 = new Response("h_0", train_me);
        pResponse_h_0->cublasHandle = cublasHandle;
        pResponse_h_0->Malloc(dim);
        responses_h_.push_back(pResponse_h_0);

        // slice out[0] into h_[1...T]
        size_t items_h_t = numel(dim);
        for (int t=1;t<seq_length_T+1;++t){
            Response* pResponse_h_t = new Response("h_"+int_to_str(t), train_me);
            pResponse_h_t->cublasHandle = cublasHandle;
            pResponse_h_t->Malloc(dim, out[0]->dataGPU + items_h_t * (t-1), out[0]->diffGPU + items_h_t * (t-1));
            responses_h_.push_back(pResponse_h_t);
        }

        // slice cont into cont_[t]
        dim = in[1]->dim;
        dim[0] = 1;
        size_t items_cont_t = numel(dim); //in[1]->sizeofitem();
        //std::cout<<"bytes_cont_t="<<bytes_cont_t<<std::endl;
        for (int t=0;t<seq_length_T;++t){
            Response* pResponse_cont_t = new Response("cont_"+int_to_str(t), false);
            pResponse_cont_t->cublasHandle = cublasHandle;
            pResponse_cont_t->Malloc(dim, in[1]->dataGPU + items_cont_t*t, NULL);
            //std::cout<<"pResponse_cont_t="<<pResponse_cont_t->dataGPU<<std::endl;
            responses_cont_.push_back(pResponse_cont_t);
        }

        // create a proxy
        dim = in[0]->dim;
        dim.erase(dim.begin());
        dim[0] *= in[0]->dim[0];
        while (dim.size()<3) dim.push_back(1);
        in0 = new Response(this->in[0]->name+"_proxy", in[0]->need_diff);
        in0->cublasHandle = cublasHandle;
        in0->Malloc(dim, in[0]->dataGPU, in[0]->diffGPU);

        // Add layer to transform all timesteps of x to the hidden state dimension.
        //     W_xc_x = W_xc * x + b_c
        pLayer_x_transform = new InnerProductLayer("x_transform", num_output*4, true, weight_lr_mult, weight_filler, weight_filler_param, bias_lr_mult, bias_filler, bias_filler_param);
        pLayer_x_transform->cudnnHandle = cudnnHandle;
        pLayer_x_transform->cublasHandle = cublasHandle;
        pLayer_x_transform->GPU = GPU;
        pLayer_x_transform->addIn(in0);
        pResponse_W_xc_x = new Response("W_xc_x", train_me);
        pResponse_W_xc_x->cublasHandle = cublasHandle;
        pLayer_x_transform->addOut(pResponse_W_xc_x);
        memoryBytes += pLayer_x_transform->Malloc(phase_);
        sub_layers.push_back(pLayer_x_transform);

        // slice W_xc_x into W_xc_x_[t]
        dim.clear();
        dim.push_back(1);
        dim.push_back(in[0]->dim[1]);
        dim.push_back(num_output*4);
        size_t items_W_xc_x_t = numel(dim);
        for (int t=0;t<seq_length_T;++t){
            Response* pResponse_W_xc_x_t = new Response("W_xc_x_"+int_to_str(t), train_me);
            pResponse_W_xc_x_t->cublasHandle = cublasHandle;
            pResponse_W_xc_x_t->Malloc(dim, pResponse_W_xc_x->dataGPU + items_W_xc_x_t*t, pResponse_W_xc_x->diffGPU  + items_W_xc_x_t*t);
            responses_W_xc_x_.push_back(pResponse_W_xc_x_t);
        }

        if (in.size()>2){
            // Add layer to transform x_static to the gate dimension.
            //     W_xc_x_static = W_xc_static * x_static
            pLayer_x_static_transform = new InnerProductLayer("W_xc_x_static", num_output*4, false, weight_lr_mult, weight_filler, weight_filler_param, bias_lr_mult, bias_filler, bias_filler_param);
            pLayer_x_static_transform->cudnnHandle = cudnnHandle;
            pLayer_x_static_transform->cublasHandle = cublasHandle;
            pLayer_x_static_transform->GPU = GPU;
            pLayer_x_static_transform ->addIn(this->in[2]);
            pResponse_W_xc_x_static = new Response("W_xc_x_static", train_me);
            pResponse_W_xc_x_static->cublasHandle = cublasHandle;
            pLayer_x_static_transform ->addOut(pResponse_W_xc_x_static);
            memoryBytes += pLayer_x_static_transform->Malloc(phase_);
            sub_layers.push_back(pLayer_x_static_transform);
        }

        dim.clear();
        dim.push_back(1);
        dim.push_back(batch_size_N);
        dim.push_back(num_output);
        // all c
        for (int t=0;t<seq_length_T+1;++t){
            Response* pResponse_c_t = new Response("c_"+int_to_str(t), train_me);
            pResponse_c_t->cublasHandle = cublasHandle;
            pResponse_c_t->Malloc(dim);
            responses_c_.push_back(pResponse_c_t);
        }

        dim.clear();
        dim.push_back(batch_size_N);
        dim.push_back(num_output);
        dim.push_back(1);
        // all h_conted_
        for (int t=0;t<seq_length_T;++t){
            Response* pResponse_h_conted_t = new Response("h_conted_"+int_to_str(t), train_me);
            pResponse_h_conted_t->cublasHandle = cublasHandle;
            pResponse_h_conted_t->Malloc(dim);
            responses_h_conted_.push_back(pResponse_h_conted_t);
        }

        reset();

        dim.clear();
        dim.push_back(1);
        dim.push_back(batch_size_N);
        dim.push_back(num_output*4);
        //responses_W_hc_h_
        for (int t=0;t<seq_length_T;++t){
            Response* pResponse_W_hc_h_t = new Response("W_hc_h_"+int_to_str(t), train_me);
            pResponse_W_hc_h_t->cublasHandle = cublasHandle;
            pResponse_W_hc_h_t->Malloc(dim);
            responses_W_hc_h_.push_back(pResponse_W_hc_h_t);
        }
        
        dim.clear();
        dim.push_back(batch_size_N);
        dim.push_back(4);
        dim.push_back(num_output);

        //responses_gate_input_
        for (int t=0;t<seq_length_T;++t){
            Response* pResponse_gate_input_t = new Response("gate_input_"+int_to_str(t), train_me);
            pResponse_gate_input_t->cublasHandle = cublasHandle;
            pResponse_gate_input_t->Malloc(dim);
            responses_gate_input_.push_back(pResponse_gate_input_t);
        }

        size_t X_bytes = sizeofStorageT * numel(dim);
        for (int t=0;t<seq_length_T;++t){
            StorageT* X_acts;
            checkCUDA(__LINE__, cudaMalloc(&X_acts, X_bytes) );
            X_acts_.push_back(X_acts);
        }
        memoryBytes += X_bytes * seq_length_T;

        // Add layers to flush the hidden state when beginning a new
        // sequence, as indicated by cont_t.
        //     h_conted_{t-1} := cont_t * h_{t-1}
        //
        // Normally, cont_t is binary (i.e., 0 or 1), so:
        //     h_conted_{t-1} := h_{t-1} if cont_t == 1
        //                       0   otherwise
        pLayer_h_conted = new ElementWiseLayer("h_conted", ElementWise_SUM, true);
        pLayer_h_conted->cudnnHandle = cudnnHandle;
        pLayer_h_conted->cublasHandle = cublasHandle;
        pLayer_h_conted->GPU = GPU;
        pLayer_h_conted->addIn(responses_h_[0]);
        pLayer_h_conted->addIn(responses_cont_[0]);
        pLayer_h_conted->addOut(responses_h_conted_[0]);
        memoryBytes += pLayer_h_conted->Malloc(phase_);
        sub_layers.push_back(pLayer_h_conted);

        // Add layer to compute
        //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
        pLayer_transform = new InnerProductLayer("transform", num_output*4, false, weight_lr_mult, weight_filler, weight_filler_param, bias_lr_mult, bias_filler, bias_filler_param);
        pLayer_transform->cudnnHandle = cudnnHandle;
        pLayer_transform->cublasHandle = cublasHandle;
        pLayer_transform->GPU = GPU;                    
        pLayer_transform->addIn(responses_h_conted_[0]);
        pLayer_transform->addOut(responses_W_hc_h_[0]);
        memoryBytes += pLayer_transform->Malloc(phase_);
        sub_layers.push_back(pLayer_transform);

        // Add the outputs of the linear transformations to compute the gate input.
        //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
        //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
        pLayer_gate_input = new ElementWiseLayer("gate_input", ElementWise_SUM);
        pLayer_gate_input->cudnnHandle = cudnnHandle;
        pLayer_gate_input->cublasHandle = cublasHandle;
        pLayer_gate_input->GPU = GPU;                    
        pLayer_gate_input->addIn(responses_W_hc_h_[0]);
        pLayer_gate_input->addIn(responses_W_xc_x_[0]);
        if (in.size()>2) {
            pLayer_gate_input->addIn(pResponse_W_xc_x_static);
        }        
        pLayer_gate_input->addOut(responses_gate_input_[0]);
        memoryBytes += pLayer_gate_input->Malloc(phase_);
        sub_layers.push_back(pLayer_gate_input);

        // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
        // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
        // Outputs: c_t, h_t
        //     [ i_t' ]
        //     [ f_t' ] := gate_input_t
        //     [ o_t' ]
        //     [ g_t' ]
        //         i_t := \sigmoid[i_t']
        //         f_t := \sigmoid[f_t']
        //         o_t := \sigmoid[o_t']
        //         g_t := \tanh[g_t']
        //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
        //         h_t := o_t .* \tanh[c_t]
        pLayer_lstm_unit = new LSTMUnitLayer("lstm_unit");
        pLayer_lstm_unit->cudnnHandle = cudnnHandle;
        pLayer_lstm_unit->cublasHandle = cublasHandle;
        pLayer_lstm_unit->GPU = GPU;                    
        pLayer_lstm_unit->addIn(responses_c_[0]);
        pLayer_lstm_unit->addIn(responses_gate_input_[0]);
        pLayer_lstm_unit->addIn(responses_cont_[0]);
        pLayer_lstm_unit->addOut(responses_c_[1]);
        pLayer_lstm_unit->addOut(responses_h_[1]);
        memoryBytes += pLayer_lstm_unit->Malloc(phase_);
        sub_layers.push_back(pLayer_lstm_unit);

        return memoryBytes;
    };

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;
        train_me = train_me && phase_ != Testing;

        std::cout<< (train_me? "* " : "  ");
        std::cout<<name;

        if (!(in.size()==2 || in.size()==3)) { std::cout<<std::endl<<"LSTMLayer #in should be 2 or 3."<<std::endl; FatalError(__LINE__); }
        if (out.size()!=1) { std::cout<<std::endl<<"LSTMLayer #out should be 1."<<std::endl; FatalError(__LINE__); }

        seq_length_T = in[0]->dim[0];
        batch_size_N = in[0]->dim[1];

        std::cout<<" T="<<seq_length_T;
        std::cout<<" N="<<batch_size_N;
        std::cout<<" num_output="<<num_output;
        std::cout<<std::endl;

        out[0]->need_diff = train_me || in[0]->need_diff; // if one of them need the grad
        if (in.size()==3) out[0]->need_diff = out[0]->need_diff || in[2]->need_diff;

        std::vector<int> dimOut(3);
        dimOut[0] = seq_length_T*batch_size_N;
        dimOut[1] = num_output;
        dimOut[2] = 1;

        out[0]->receptive_field = in[0]->receptive_field;
        out[0]->receptive_gap = in[0]->receptive_gap;
        out[0]->receptive_offset = in[0]->receptive_offset;
        memoryBytes += out[0]->Malloc(dimOut);

        memoryBytes += FillUnrolledNet(phase_);

        return memoryBytes;
    };

    void reset(){
        checkCUDA(__LINE__,cudaMemset(responses_c_[0]->dataGPU, 0, responses_c_[0]->numBytes()));        
        checkCUDA(__LINE__,cudaMemset(responses_h_[0]->dataGPU, 0, responses_h_[0]->numBytes()));
    };

    void forward(Phase phase_){

        // copy c[T] to c[0]
        checkCUDA(__LINE__,cudaMemcpy(responses_c_[responses_c_.size()-1]->dataGPU, responses_c_[0]->dataGPU, responses_c_[0]->numBytes(), cudaMemcpyDeviceToDevice));

        // copy h[T] to h[0]
        checkCUDA(__LINE__,cudaMemcpy(responses_h_[responses_h_.size()-1]->dataGPU, responses_h_[0]->dataGPU, responses_h_[0]->numBytes(), cudaMemcpyDeviceToDevice));

        ComputeT avg;

        if (debug_mode){
            avg = in0->ameanData();
            std::cout<< in0->name << ".data_amean: " << avg ;
            in0->checkNaN();
            std::cout<< std::endl;
        }
        pLayer_x_transform->forward(phase_);
        if (debug_mode){
            std::cout<< "Layer: W_xc_x = W_xc * x + b_c      ";
            avg = pLayer_x_transform->ameanWeightData(); if (avg!=-1) std::cout<<" weight.data: "<< avg;
            pLayer_x_transform->checkNaNWeight();
            avg = pLayer_x_transform->ameanBiasData();   if (avg!=-1) std::cout<<" bias.data: "<< avg;            
            pLayer_x_transform->checkNaNBias();
            std::cout<< std::endl;
        }
        if (debug_mode){
            avg = pResponse_W_xc_x->ameanData();
            std::cout<< pResponse_W_xc_x->name << ".data_amean: " << avg ;
            pResponse_W_xc_x->checkNaN();
            std::cout<< std::endl;
        }

        if (in.size()>2){
            pLayer_x_static_transform->forward(phase_);
        }

        for (int t=0;t<seq_length_T;++t){
            
            if (debug_mode)
                std::cout<<"====================================================================================================================================="<<std::endl;

            // Add layers to flush the hidden state when beginning a new
            // sequence, as indicated by cont_t.
            //     h_conted_{t-1} := cont_t * h_{t-1}
            //
            // Normally, cont_t is binary (i.e., 0 or 1), so:
            //     h_conted_{t-1} := h_{t-1} if cont_t == 1, and 0 otherwise

            if (debug_mode){
                avg = responses_h_[t]->ameanData();
                std::cout<< responses_h_[t]->name << ".data_amean: " << avg ;
                responses_h_[t]->checkNaN();
                std::cout<< std::endl;
            }

            if (debug_mode){
                avg = responses_cont_[t]->ameanData();
                std::cout<< responses_cont_[t]->name << ".data_amean: " << avg ;
                responses_cont_[t]->checkNaN();
                std::cout<< std::endl;
            }

            pLayer_h_conted->in [0]=responses_h_[t];
            pLayer_h_conted->in [1]=responses_cont_[t];
            pLayer_h_conted->out[0]=responses_h_conted_[t];
            pLayer_h_conted->forward(phase_);

            if (debug_mode){
                std::cout<< "Layer: h_conted_{t-1} := cont_t * h_{t-1}"<< std::endl;
            }

            if (debug_mode){
                avg = responses_h_conted_[t]->ameanData();
                std::cout<< responses_h_conted_[t]->name << ".data_amean: " << avg ;
                responses_h_conted_[t]->checkNaN();
                std::cout<< std::endl;
            }

            // Add layer to compute
            //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
            pLayer_transform->in[0]=responses_h_conted_[t];
            pLayer_transform->out[0]=responses_W_hc_h_[t];
            pLayer_transform->forward(phase_);

            if (debug_mode){
                std::cout<< "Layer: W_hc_h_{t-1} := W_hc * h_conted_{t-1}";
                std::cout<<"  bias_numel="<<pLayer_transform->bias_numel<< "  ";
                avg = pLayer_transform->ameanWeightData(); if (avg!=-1) std::cout<<" weight.data: "<< avg;
                pLayer_transform->checkNaNWeight();                
                std::cout<< std::endl;
            }

            if (debug_mode){
                avg = responses_W_hc_h_[t]->ameanData();
                std::cout<< responses_W_hc_h_[t]->name << ".data_amean: " << avg ;
                responses_W_hc_h_[t]->checkNaN();
                std::cout<< std::endl;
            }

            if (debug_mode){
                avg = responses_W_xc_x_[t]->ameanData();
                std::cout<< responses_W_xc_x_[t]->name << ".data_amean: " << avg ;
                responses_W_xc_x_[t]->checkNaN();
                std::cout<< std::endl;
            }            

            // Add the outputs of the linear transformations to compute the gate input.
            //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
            //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
            pLayer_gate_input->in[0]=responses_W_hc_h_[t];
            pLayer_gate_input->in[1]=responses_W_xc_x_[t];
            pLayer_gate_input->out[0]=responses_gate_input_[t];
            pLayer_gate_input->forward(phase_);

            if (debug_mode){
                std::cout<< "Layer: gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c"<< std::endl;
            }

            if (debug_mode){
                avg = responses_gate_input_[t]->ameanData();
                std::cout<< responses_gate_input_[t]->name << ".data_amean: " << avg ;
                responses_gate_input_[t]->checkNaN();
                std::cout<< std::endl;
            }

            if (debug_mode){
                avg = responses_c_[t]->ameanData();
                std::cout<< responses_c_[t]->name << ".data_amean: " << avg ;
                responses_c_[t]->checkNaN();
                std::cout<< std::endl;
            }

            // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
            // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
            // Outputs: c_t, h_t
            //     [ i_t' ]
            //     [ f_t' ] := gate_input_t
            //     [ o_t' ]
            //     [ g_t' ]
            //         i_t := \sigmoid[i_t']
            //         f_t := \sigmoid[f_t']
            //         o_t := \sigmoid[o_t']
            //         g_t := \tanh[g_t']
            //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
            //         h_t := o_t .* \tanh[c_t]
            pLayer_lstm_unit->in[0]=responses_c_[t];
            pLayer_lstm_unit->in[1]=responses_gate_input_[t];
            pLayer_lstm_unit->in[2]=responses_cont_[t];
            pLayer_lstm_unit->out[0]=responses_c_[t+1];
            pLayer_lstm_unit->out[1]=responses_h_[t+1];
            pLayer_lstm_unit->X_acts = X_acts_[t];
            pLayer_lstm_unit->forward(phase_);

            if (debug_mode){
                std::cout<< "Layer: LSTMUnit"<< std::endl;
            }            

            if (debug_mode){
                avg = responses_c_[t+1]->ameanData();
                std::cout<< responses_c_[t+1]->name << ".data_amean: " << avg ;
                responses_c_[t+1]->checkNaN();
                std::cout<< std::endl;
            }

            if (debug_mode){
                avg = responses_h_[t+1]->ameanData();
                std::cout<< responses_h_[t+1]->name << ".data_amean: " << avg ;
                responses_h_[t+1]->checkNaN();
                std::cout<< std::endl;
            }            

        }
    };

    void backward(Phase phase_){
        if (in.size()>2) pResponse_W_xc_x_static->clearDiff();
        pResponse_W_xc_x->clearDiff();
        for (int r=0;r<responses_W_xc_x_.size();++r) responses_W_xc_x_[r]->clearDiff();
        for (int r=0;r<responses_cont_.size();++r) responses_cont_[r]->clearDiff();
        for (int r=0;r<responses_c_.size();++r) responses_c_[r]->clearDiff();
        for (int r=0;r<responses_h_.size();++r) responses_h_[r]->clearDiff();
        for (int r=0;r<responses_h_conted_.size();++r) responses_h_conted_[r]->clearDiff();
        for (int r=0;r<responses_W_hc_h_.size();++r) responses_W_hc_h_[r]->clearDiff();
        for (int r=0;r<responses_gate_input_.size();++r) responses_gate_input_[r]->clearDiff();

        if (in.size()>2) pLayer_x_static_transform->clearDiff();
        pLayer_x_transform->clearDiff();
        pLayer_h_conted->clearDiff();
        pLayer_transform->clearDiff();
        pLayer_gate_input->clearDiff();
        pLayer_lstm_unit->clearDiff();

        for (int t=seq_length_T-1;t>=0;--t){
            pLayer_lstm_unit->in[0]=responses_c_[t];
            pLayer_lstm_unit->in[1]=responses_gate_input_[t];
            pLayer_lstm_unit->in[2]=responses_cont_[t];
            pLayer_lstm_unit->out[0]=responses_c_[t+1];
            pLayer_lstm_unit->out[1]=responses_h_[t+1];
            pLayer_lstm_unit->X_acts = X_acts_[t];
            pLayer_lstm_unit->backward(phase_);

            pLayer_gate_input->in[0]=responses_W_hc_h_[t];
            pLayer_gate_input->in[1]=responses_W_xc_x_[t];
            pLayer_gate_input->out[0]=responses_gate_input_[t];
            pLayer_gate_input->backward(phase_);

            pLayer_transform->in[0]=responses_h_conted_[t];
            pLayer_transform->out[0]=responses_W_hc_h_[t];
            pLayer_transform->backward(phase_);            

            pLayer_h_conted->in [0]=responses_h_[t];
            pLayer_h_conted->in [1]=responses_cont_[t];
            pLayer_h_conted->out[0]=responses_h_conted_[t];
            pLayer_h_conted->backward(phase_);
        }
        if (in.size()>2){
            pLayer_x_static_transform->backward(phase_);
        }
        pLayer_x_transform->backward(phase_);
    };

    ~LSTMLayer(){
        delete pLayer_x_transform;
        if (in.size()>2)
            delete pLayer_x_static_transform;
        delete pLayer_h_conted;
        delete pLayer_transform;
        delete pLayer_gate_input;
        delete pLayer_lstm_unit;

        delete in0;
        delete pResponse_W_xc_x_static;
        delete pResponse_W_xc_x;
        for (int r=0;r<responses_W_xc_x_.size();++r) delete responses_W_xc_x_[r];
        for (int r=0;r<responses_cont_.size();++r) delete responses_cont_[r];
        for (int r=0;r<responses_c_.size();++r) delete responses_c_[r];
        for (int r=0;r<responses_h_.size();++r) delete responses_h_[r];
        for (int r=0;r<responses_h_conted_.size();++r) delete responses_h_conted_[r];
        for (int r=0;r<responses_W_hc_h_.size();++r) delete responses_W_hc_h_[r];
        for (int r=0;r<responses_gate_input_.size();++r) delete responses_gate_input_[r];

        for (int t=0;t<X_acts_.size();++t)  checkCUDA(__LINE__, cudaFree(X_acts_[t]));
    };
};

class BatchNormalizationLayer : public Layer{
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
    cudnnBatchNormMode_t mode;
    double epsilon;

    unsigned int numForwardTrainingPasses;

    // NOTE: we use weight and bias in place of scale and bias so we don't modify the solver
    // StorageT* bnScale;               => weight_dataGPU
    // StorageT* bnBias;                => bias_dataGPU
    StorageT* resultRunningMean;
    StorageT* resultRunningInvVariance;
    StorageT* resultSaveMean;
    StorageT* resultSaveInvVariance;
    // StorageT* resultBnScaleDiff;     => weight_diffGPU
    // StorageT* resultBnBiasDiff;      => bias_diffGPU
public:

    BatchNormalizationLayer(JSON* json){
        if (in.size() != out.size()){ std::cout << "BatchNormalizationLayer " << name << " needs same number of input and output responses" << std::endl; FatalError(__LINE__); }
        for (int i = 1; i < in.size(); ++i){
            if (!same_dim(in[0]->dim, in[i]->dim)){ std::cout <<"BatchNormalizationLayer" << name << " requires same dim for each input" << std::endl; FatalError(__LINE__); }
        }

        SetOrDie(json, name)
        SetValue(json, phase,                   TrainingTesting)
        SetValue(json, train_me,                true)
        SetValue(json, mode,                    CUDNN_BATCHNORM_PER_ACTIVATION)
        SetValue(json, epsilon,                 CUDNN_BN_MIN_EPSILON)
        SetValue(json, weight_lr_mult,          1.0)
        SetValue(json, weight_filler,           Constant)
        SetValue(json, weight_filler_param,     1)
        SetValue(json, bias_lr_mult,            1.0)
        SetValue(json, bias_filler,             Constant)
        SetValue(json, bias_filler_param,       0.001)

        numForwardTrainingPasses = 0;
    };
    // BatchNormalizationLayer(attributes)

    size_t Malloc(Phase phase_){
        size_t memoryBytes = 0;

        train_me = train_me && phase_ != Testing;
        std::cout << (train_me? "* " : "  ");
        std::cout << name;

        // NOTE: calculated by cudnnDeriveBNTensorDescriptor
        // TODO: delete this section
        // std::vector<int> dim;
        // std::vector<int> stride;

        // dim.resize(in[i]->dim.size());
        // stride.resize(in[i]->dim.size());

        // // There is one statistic per batch
        // dim[0] = 1;

        // // In Spatial mode, we have 1xCx1x1 tensor for batch normalization
        // // statistic in 2D images and 1xCx1x1x1 in 3D (not documented)
        // if (mode == CUDNN_BATCHNORM_SPATIAL){
        //     for (int j = 2; j < dim.size(); ++j){
        //         dim[j] = 1;
        //     }
        // }

        // // Calculate appropriate stride
        // stride[dim.size() - 1] = 1;
        // for (int d = dim.size() - 2; d >= 0; --d){
        //     stride[d] = stride[d + 1] *  dim[d + 1];
        // }

        // checkCUDNN(__LINE__, cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc) );

        // checkCUDNN(__LINE__, cudnnSetTensorNdDescriptor(
        //     bnScaleBiasMeanVarDesc,
        //     CUDNNStorageT,
        //     dim.size(),
        //     &dim[0],
        //     &stride[0]) );

        // We check that the dimensions of all in responses are the same, so this is OK
        std::vector<int> dim(in[0]->dim);

        // There is one statistic per batch
        dim[0] = 1;

        // In Spatial mode, we have 1xCx1x1 tensor for batch normalization
        // statistic in 2D images and 1xCx1x1x1 in 3D
        if (mode == CUDNN_BATCHNORM_SPATIAL){
            for (int j = 2; j < dim.size(); ++j){
                dim[j] = 1;
            }
        }

        weight_dim = dim;
        bias_dim = dim;

        // We check that the dimensions of all in responses are the same, so this is OK
        checkCUDNN(__LINE__, cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc) );
        checkCUDNN(__LINE__, cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc, in[0]->getDesc(), mode) );
        
        // These should be the same, but for clarity
        weight_numel = numel(weight_dim);
        bias_numel = numel(bias_dim);

        size_t sizeofBNScaleBiasMeanVar = numel(dim) * sizeofStorageT;

        std::cout << " 6x bnScaleBiasMeanVar"; veciPrint(dim);
        checkCUDA(__LINE__, cudaMalloc( &weight_dataGPU,            sizeofBNScaleBiasMeanVar) );
        checkCUDA(__LINE__, cudaMalloc( &bias_dataGPU,              sizeofBNScaleBiasMeanVar) );
        memoryBytes += sizeofBNScaleBiasMeanVar * 2;

        // TODO: We could allocate this as a contiguous block of memory
        // We want to keep a history of the running statistic for each in response
        checkCUDA(__LINE__, cudaMalloc( &resultRunningMean,         sizeofBNScaleBiasMeanVar * in.size()) );
        checkCUDA(__LINE__, cudaMalloc( &resultRunningInvVariance,  sizeofBNScaleBiasMeanVar * in.size()) );
        checkCUDA(__LINE__, cudaMalloc( &resultSaveMean,            sizeofBNScaleBiasMeanVar * in.size()) );
        checkCUDA(__LINE__, cudaMalloc( &resultSaveInvVariance,     sizeofBNScaleBiasMeanVar * in.size()) );
        memoryBytes += sizeofBNScaleBiasMeanVar * 4 * in.size();

        for (int i = 0; i < out.size(); ++i){
            out[i]->need_diff = train_me || in[i]->need_diff; // if one of them need the grad
            out[i]->receptive_field = in[i]->receptive_field;
            out[i]->receptive_gap = in[i]->receptive_gap;
            out[i]->receptive_offset = in[i]->receptive_offset;
            memoryBytes += out[i]->Malloc(in[i]->dim);
        }
        
        return memoryBytes;
    };

    // TODO: 'Much higher performance for HW-packed tensors for both x and y'
    // TODO: We should keep running tally of where we are in dataGPU and diffGPU
    // TODO: Implement groups?
    void forward(Phase phase_){    
        for (int i = 0; i < in.size(); ++i){
            if (phase_ == Training){

                int weight_index = i * weight_numel;
                int bias_index = i * bias_numel;

                checkCUDNN(__LINE__, cudnnBatchNormalizationForwardTraining(
                    cudnnHandle,
                    mode,
                    one,
                    zero,
                    in[i]->getDesc(),
                    in[i]->dataGPU,
                    out[i]->getDesc(),
                    out[i]->dataGPU,
                    bnScaleBiasMeanVarDesc,
                    weight_dataGPU,
                    bias_dataGPU,
                    1. / (1 + numForwardTrainingPasses),
                    resultRunningMean + weight_index,
                    resultRunningInvVariance + bias_index,
                    epsilon,
                    resultSaveMean + weight_index,
                    resultSaveInvVariance + bias_index) );

                // We want the next in response to use the running
                // mean/variance we have just computed
                if (i < in.size() - 1) {                    
                    checkCUDA(__LINE__, cudaMemcpy(resultRunningMean + weight_index + weight_numel, resultRunningMean + weight_index, sizeofStorageT * weight_numel, cudaMemcpyDeviceToDevice));
                    checkCUDA(__LINE__, cudaMemcpy(resultRunningInvVariance + bias_index + bias_numel, resultRunningInvVariance + bias_index, sizeofStorageT * bias_numel, cudaMemcpyDeviceToDevice));
                    checkCUDA(__LINE__, cudaMemcpy(resultSaveMean + weight_index + weight_numel, resultSaveMean + weight_index, sizeofStorageT * weight_numel, cudaMemcpyDeviceToDevice));
                    checkCUDA(__LINE__, cudaMemcpy(resultSaveInvVariance + bias_index + bias_numel, resultSaveInvVariance + bias_index, sizeofStorageT * bias_numel, cudaMemcpyDeviceToDevice));
                }

                numForwardTrainingPasses++;
            } else{
                checkCUDNN(__LINE__, cudnnBatchNormalizationForwardTraining(
                    cudnnHandle,
                    mode,
                    one,
                    zero,
                    in[i]->getDesc(),
                    in[i]->dataGPU,
                    out[i]->getDesc(),
                    out[i]->dataGPU,
                    bnScaleBiasMeanVarDesc,
                    weight_dataGPU,
                    bias_dataGPU,
                    1,
                    NULL,
                    NULL,
                    epsilon,
                    NULL,
                    NULL) );
            }
            // TODO: Filed bug report with NVIDIA
            //     checkCUDNN(__LINE__, cudnnBatchNormalizationForwardInference(
            //         cudnnHandle,
            //         mode,
            //         one,
            //         zero,
            //         in[i]->getDesc(),
            //         in[i]->dataGPU,
            //         out[i]->getDesc(),
            //         out[i]->dataGPU,
            //         bnScaleBiasMeanVarDesc,
            //         weight_dataGPU,
            //         bias_dataGPU,
            //         resultRunningMean,
            //         resultRunningInvVariance,
            //         epsilon) );
            // }
        }
    };

    // TODO: 'Much higher performance when HW-packed tensors are used for all of x, dy, dx'
    // TODO: We should keep running tally of where we are in dataGPU and diffGPU
    // TODO: Implement groups?
    void backward(Phase phase_){

        int last_weight_index = (in.size() - 1) * weight_numel;
        int last_bias_index = (in.size() - 1) * bias_numel;

        for (int i = 0; i < in.size(); ++i){

            int weight_index = i * weight_numel;
            int bias_index = i * bias_numel;

            checkCUDNN(__LINE__, cudnnBatchNormalizationBackward(
                cudnnHandle,
                mode,
                one,
                zero,
                one,
                zero,
                in[0]->getDesc(),
                in[0]->dataGPU,
                out[0]->getDesc(),
                out[0]->diffGPU,
                in[0]->getDesc(),
                in[0]->diffGPU,
                bnScaleBiasMeanVarDesc,
                weight_dataGPU,
                weight_diffGPU,
                bias_diffGPU,
                epsilon,
                resultSaveMean + weight_index,
                resultSaveInvVariance + bias_index) );

        // Copy running statistic from last in response to earlier in
        // responses. Technically, we only need to copy to the first one since
        // the next forward pass should take care of propagating new running
        // values
            if (i < in.size() - 1) {                    
                checkCUDA(__LINE__, cudaMemcpy(resultRunningMean + weight_index, resultRunningMean + last_weight_index, sizeofStorageT * weight_numel, cudaMemcpyDeviceToDevice));
                checkCUDA(__LINE__, cudaMemcpy(resultRunningInvVariance + bias_index, resultRunningInvVariance + last_bias_index, sizeofStorageT * weight_numel, cudaMemcpyDeviceToDevice));
                checkCUDA(__LINE__, cudaMemcpy(resultSaveMean + weight_index, resultSaveMean + last_weight_index, sizeofStorageT * weight_numel, cudaMemcpyDeviceToDevice));
                checkCUDA(__LINE__, cudaMemcpy(resultSaveInvVariance + bias_index, resultSaveInvVariance + last_bias_index, sizeofStorageT * weight_numel, cudaMemcpyDeviceToDevice));
            }
        }
    };

    ~BatchNormalizationLayer(){
        checkCUDNN(__LINE__, cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc) );

        if (resultRunningMean != NULL)          checkCUDA(__LINE__, cudaFree(resultRunningMean) );
        if (resultRunningInvVariance != NULL)   checkCUDA(__LINE__, cudaFree(resultRunningInvVariance) );
        if (resultSaveMean != NULL)             checkCUDA(__LINE__, cudaFree(resultSaveMean) );
        if (resultSaveInvVariance != NULL)      checkCUDA(__LINE__, cudaFree(resultSaveInvVariance) );
    };
};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Add your new layers here
//////////////////////////////////////////////////////////////////////////////////////////////////

#include "match.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////
// Net
//////////////////////////////////////////////////////////////////////////////////////////////////

class Net{
public:
    Phase phase;
    std::vector<Layer*> layers;
    std::vector<Response*> responses;
    std::vector<LossLayer*> loss_layers;
    int GPU;
    bool debug_mode;
    int train_iter;
    int test_iter;
    int display_iter;

    cudnnHandle_t cudnnHandle;
    cublasHandle_t cublasHandle;

    void init(JSON* architecture_obj){
        checkCUDA(__LINE__,cudaSetDevice(GPU));

        checkCUDNN(__LINE__,cudnnCreate(&cudnnHandle) );
        checkCUBLAS(__LINE__, cublasCreate(&cublasHandle) );

        std::vector<SequenceGenerationLayer*> sequence_layers;

        for (int l=0;l<architecture_obj->array.size();++l){

            JSON* p = (JSON*)(architecture_obj->array[l]);

            std::string type = p->member["type"]->returnString();

            Layer* pLayer;
            Response* pResponse;

                 if (0==type.compare("MemoryData"))     pLayer = new MemoryDataLayer(p);
            else if (0==type.compare("DiskData")){
                uint8_t fpTypeid = readTypeID(p->member["file_data"]->returnString());
                     if (fpTypeid==typeID(typeid(half)))        pLayer = new DiskDataLayer<half>(p);
                else if (fpTypeid==typeID(typeid(float)))       pLayer = new DiskDataLayer<float>(p);
                else if (fpTypeid==typeID(typeid(double)))      pLayer = new DiskDataLayer<double>(p);
                else if (fpTypeid==typeID(typeid(uint8_t)))     pLayer = new DiskDataLayer<uint8_t>(p);
                else if (fpTypeid==typeID(typeid(uint16_t)))    pLayer = new DiskDataLayer<uint16_t>(p);
                else if (fpTypeid==typeID(typeid(uint32_t)))    pLayer = new DiskDataLayer<uint32_t>(p);
                else if (fpTypeid==typeID(typeid(uint64_t)))    pLayer = new DiskDataLayer<uint64_t>(p);
                else if (fpTypeid==typeID(typeid(int8_t)))      pLayer = new DiskDataLayer<int8_t>(p);
                else if (fpTypeid==typeID(typeid(int16_t)))     pLayer = new DiskDataLayer<int16_t>(p);
                else if (fpTypeid==typeID(typeid(int32_t)))     pLayer = new DiskDataLayer<int32_t>(p);
                else if (fpTypeid==typeID(typeid(int64_t)))     pLayer = new DiskDataLayer<int64_t>(p);
                else if (fpTypeid==typeID(typeid(char)))        pLayer = new DiskDataLayer<char>(p);
                else if (fpTypeid==typeID(typeid(bool)))        pLayer = new DiskDataLayer<bool>(p);
            }
#if USE_OPENCV
            else if (0==type.compare("ImageData"))              pLayer = new ImageDataLayer(p);
#endif            
            else if (0==type.compare("MatchData"))              pLayer = new MatchDataLayer<float>(p);
            else if (0==type.compare("ElementWise"))            pLayer = new ElementWiseLayer(p);
            else if (0==type.compare("Concat"))                 pLayer = new ConcatLayer(p);
            else if (0==type.compare("Convolution"))            pLayer = new ConvolutionLayer(p);
            else if (0==type.compare("Deconvolution"))          pLayer = new DeconvolutionLayer(p);
            else if (0==type.compare("Reshape"))                pLayer = new ReshapeLayer(p);
            else if (0==type.compare("InnerProduct"))           pLayer = new InnerProductLayer(p);
            else if (0==type.compare("Pooling"))                pLayer = new PoolingLayer(p);
            else if (0==type.compare("Dropout"))                pLayer = new DropoutLayer(p);
            else if (0==type.compare("Activation"))             pLayer = new ActivationLayer(p);
            else if (0==type.compare("LRN"))                    pLayer = new LRNLayer(p);
            else if (0==type.compare("Softmax"))                pLayer = new SoftmaxLayer(p);
            else if (0==type.compare("ROI"))                    pLayer = new ROILayer(p);
            else if (0==type.compare("ROIPooling"))             pLayer = new ROIPoolingLayer(p);
            else if (0==type.compare("Tensor"))                 pLayer = new TensorLayer(p);
            else if (0==type.compare("Constant"))               pLayer = new ConstantLayer(p);
            else if (0==type.compare("PlaceHolderData"))        pLayer = new PlaceHolderDataLayer(p);
            else if (0==type.compare("LSTM"))                   pLayer = new LSTMLayer(p);
            else if (0==type.compare("SequenceGeneration"))    {pLayer = new SequenceGenerationLayer(p); sequence_layers.push_back((SequenceGenerationLayer*)pLayer);   }
            else if (0==type.compare("BatchNormalization"))     pLayer = new BatchNormalizationLayer(p);
            else if (0==type.compare("Loss"))                  {pLayer = new LossLayer(p); loss_layers.push_back((LossLayer*)pLayer);   }
            else { std::cout<<"ERROR: recognizable layer in JSON file: "<<type<<std::endl; FatalError(__LINE__);};

            pLayer->cudnnHandle = cudnnHandle;
            pLayer->cublasHandle = cublasHandle;
            pLayer->GPU = GPU;

            addLayer(pLayer);

            if (p->member.find("out") != p->member.end()){
                std::vector<std::string> out = p->member["out"]->returnStringVector();
                for (int i=0;i<out.size();i++){
                    pResponse = getResponse(out[i]);
                    if (pResponse==NULL){
                        pResponse = addResponse(new Response(out[i]));
                        pResponse->cublasHandle = cublasHandle;
                    }
                    pLayer->addOut(pResponse);
                }
            }

            if (p->member.find("in") != p->member.end()){
                std::vector<std::string> in = p->member["in"]->returnStringVector();
                for (int i=0;i<in.size();i++){
                    pResponse = getResponse(in[i]);
                    if (pResponse==NULL){
                        pResponse = addResponse(new Response(in[i]));
                        pResponse->cublasHandle = cublasHandle;
                    }
                    pLayer->addIn(pResponse);
                }
            }
        }

        for (int l=0;l<sequence_layers.size();++l){
            sequence_layers[l]->resultResponse = getResponse(sequence_layers[l]->result);
        }
    };


    Net(std::string filename){
        JSON* test_obj = new JSON;
        JSON* architecture_obj = new JSON;
        parseNetworkJSON(filename, NULL, test_obj, architecture_obj);
        SetValue(test_obj, GPU,             0)
        SetValue(test_obj, debug_mode,      false)
        SetValue(test_obj, display_iter,    1)

        init(architecture_obj);

        delete test_obj;
        delete architecture_obj;
    };

    Net(JSON* architecture_obj, int GPU_ = 0): GPU(GPU_){
        init(architecture_obj);
    };

    ~Net(){
        checkCUDA(__LINE__,cudaSetDevice(GPU));

        for (int i=0;i<layers.size();++i){
            delete layers[i];
        }
        for (int i=0;i<responses.size();++i){
            delete responses[i];
        }
        checkCUDNN(__LINE__,cudnnDestroy(cudnnHandle) );
        checkCUBLAS(__LINE__, cublasDestroy(cublasHandle) );
    };

    Layer* getLayer(std::string name){
        for (int l=0; l<layers.size();++l){
            if (layers[l]->name == name){
                return layers[l];
            }
        }
        return NULL;
    };

    Response* getResponse(std::string name){
        for (int r=0; r<responses.size();++r){
            if (responses[r]->name == name){
                return responses[r];
            }
        }
        return NULL;
    };

    Layer* addLayer(Layer* pLayer){
        layers.push_back(pLayer);
        return pLayer;
    };

    Response* addResponse(Response* pResponse){
        responses.push_back(pResponse);
        return pResponse;
    };

    void randInit(){
        checkCUDA(__LINE__,cudaSetDevice(GPU));
        for (int l=0; l<layers.size();++l){
            layers[l]->randInit();
        }
    };

    void loadWeights(std::vector<Tensor<StorageT>*> weights, bool diff=false){
        checkCUDA(__LINE__,cudaSetDevice(GPU));
        // let the layers find their weights based on their names
        for (int l=0; l<layers.size();++l){
            layers[l]->setWeights(weights);
            if (diff) layers[l]->setDiffs(weights);
        }
    };

    void loadWeights(std::string filename, bool diff=false){
        std::cout<< "====================================================================================================================================="<<std::endl;

        std::vector<Tensor<StorageT>*> weights = readTensors<StorageT>(filename);
        loadWeights(weights, diff);

        // release memory for the weights
        for (int i=0; i<weights.size();++i){
            delete weights[i];
        }
    };

    void saveWeights(std::string filename, bool diff=false){
        FILE* fp = fopen(filename.c_str(),"wb");
        while (fp==NULL) {
            std::cerr<<"Net::saveWeights: fail to open file "<<filename<<". Please provide it first. Will retry after 5 seconds."<<std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));
            fp = fopen(filename.c_str(),"wb");
        }

        for (int l=0; l<layers.size();++l){
            layers[l]->saveWeights(fp);
            if (diff) layers[l]->saveDiffs(fp);
        }
        fclose(fp);
    };

    size_t Malloc(Phase phase_ = Testing){
        checkCUDA(__LINE__,cudaSetDevice(GPU));

        phase = phase_;

        std::cout<< "====================================================================================================================================="<<std::endl;
        std::cout<< "  Layers:                                                                        Responses:                                          "<<std::endl;
        std::cout<< "====================================================================================================================================="<<std::endl;

        size_t memoryBytes = 0;

        for (int l=0;l<layers.size();++l){
            memoryBytes += layers[l]->Malloc(phase);
        }

        std::cout<< "====================================================================================================================================="<<std::endl;
        std::cout<< "GPU " << GPU << ": Total GPU memory: ";    memorySizePrint(memoryBytes);   std::cout<<std::endl;

        return memoryBytes;
    };

    void forward(){
        for (int l=0; l<layers.size();++l){
            if (layers[l]->phase == phase || layers[l]->phase == TrainingTesting){
                if (debug_mode){
                    std::cout<<"[Forward] Layer["<<l<<"] "<<layers[l]->name;
                    ComputeT avg;
                    avg = layers[l]->ameanWeightData(); if (avg!=-1) std::cout<<" weight.data: "<< avg;
                    avg = layers[l]->ameanBiasData();   if (avg!=-1) std::cout<<" bias.data: "<< avg;
                    tic();
                }

                layers[l]->forward(phase);

                if (debug_mode){
                    checkCUDA(__LINE__,cudaDeviceSynchronize()); checkCUDA(__LINE__,cudaGetLastError());
                    if (layers[l]->out.size()>0){
                        for (size_t o=0; o<layers[l]->out.size();++o){
                            ComputeT avg = layers[l]->out[o]->ameanData();
                            if (avg!=-1) std::cout<<" out[" << o << "].data("<< layers[l]->out[o]->name << "): " << avg;
                            layers[l]->out[o]->checkNaN();
                        }
                    }
                    std::cout<<std::endl; toc();
                }
            }
        }
    };

    void backward(){
        for (int r=0;r<responses.size();++r){
            responses[r]->clearDiff();
        }

        for (int l=layers.size()-1;l>=0; --l){
            if (layers[l]->phase == phase || layers[l]->phase == TrainingTesting){

                if (debug_mode){
                    std::cout<<"[Backward] Layer["<<l<<"] "<<layers[l]->name;
                    tic();
                }

                layers[l]->backward(phase);

                if (debug_mode){
                    checkCUDA(__LINE__,cudaDeviceSynchronize()); checkCUDA(__LINE__,cudaGetLastError());
                    ComputeT avg;
                    avg = layers[l]->ameanWeightDiff(); if (avg!=-1) std::cout<<" weight.diff: "<<avg;
                    avg = layers[l]->ameanBiasDiff();   if (avg!=-1) std::cout<<" bias.diff: "<<avg;

                    if (layers[l]->in.size()>0){
                        for (size_t i=0;i<layers[l]->in.size();++i){
                            avg = layers[l]->in[i]->ameanDiff(); if (avg!=-1) std::cout<<" in[" << i << "].diff(" << layers[l]->in[i]->name << "): "<<avg;
                        }
                    }
                    std::cout<<std::endl; toc();
                }
            }
        }
    };

    void update(){
        for (int l=0; l<layers.size();++l){
            layers[l]->update();
        }
    };

    void resetLoss(){
        for (int l=0; l<loss_layers.size();++l){
            loss_layers[l]->result = ComputeT(0);
            loss_layers[l]->loss   = ComputeT(0);
        }
    };

    void eval(bool sync){
        checkCUDA(__LINE__,cudaSetDevice(GPU));
        for (int l=0; l<loss_layers.size();++l){
            if (loss_layers[l]->phase == phase || loss_layers[l]->phase == TrainingTesting)
                loss_layers[l]->eval();
        }
        if(sync) checkCUDA(__LINE__,cudaDeviceSynchronize());
    };

    void stepTest(bool sync){
        checkCUDA(__LINE__,cudaSetDevice(GPU));

        resetLoss();
        for (int i=0; i < test_iter; ++i){
            forward();
            eval(false);
        }
        for (int l=0; l<loss_layers.size();++l){
            loss_layers[l]->result /= test_iter;
            loss_layers[l]->loss   /= test_iter;
        }

        if(sync) checkCUDA(__LINE__,cudaDeviceSynchronize());
    };


    void stepTrain(bool sync){
        checkCUDA(__LINE__,cudaSetDevice(GPU));

        update();

        resetLoss();
        for (int l=0; l<layers.size();++l){
            layers[l]->clearDiff();
        }

        for (int i=0; i < train_iter; ++i){
            forward();
            backward();
        }

        for (int l=0; l<loss_layers.size();++l){
            loss_layers[l]->result /= train_iter;
            loss_layers[l]->loss   /= train_iter;
        }

        if(sync) checkCUDA(__LINE__,cudaDeviceSynchronize());
    };


    void getTopActivations(std::string dataResponseName, std::vector<std::string> responseNames, std::vector<std::vector<int> > responseChannels, std::string saveFilePrefix, int topK, int maxIterations){

        phase = Training;

        DataLayer* pDataLayer = NULL;
        for (int l=0; l<layers.size();++l){
            if (layers[l]->phase == phase || layers[l]->phase == TrainingTesting){
                if (layers[l]->isDataLayer()){
                    pDataLayer = (DataLayer*) layers[l];
                    break;
                }
            }
        }
        if (pDataLayer==NULL) { std::cerr<<"No data layer."<<std::endl; FatalError(__LINE__);};

        Response* rData = getResponse(dataResponseName);

        std::vector<std::vector<std::vector<Tensor<StorageT>*> > > data;
        std::vector<std::vector<std::vector<ComputeT > > > scores;
        std::vector<std::vector<ComputeT > >  scoresLowest;
        data.resize(responseChannels.size());
        scores.resize(responseChannels.size());
        scoresLowest.resize(responseChannels.size());
        for(int i=0;i<responseNames.size();++i){
            data[i].resize(responseChannels[i].size());
            scores[i].resize(responseChannels[i].size());
            scoresLowest[i].resize(responseChannels[i].size());
        }


        int dataChannels = rData->dim[1];
        Tensor<StorageT>* rdataTensor = new Tensor<StorageT>(rData->dim);

        int iter = 0;

        while(pDataLayer->epoch == 0 && iter < maxIterations){
            resetLoss();
            forward();
            eval(false);

            // display
            std::cout << "Iteration " << iter << "  ";
            for (int i=0;i<loss_layers.size();++i){
                if (loss_layers[i]->phase == phase || loss_layers[i]->phase == TrainingTesting){
                    loss_layers[i]->display();
                }
            }
            std::cout << std::endl;

            rdataTensor->readGPU(rData->dataGPU);

            for(int i=0;i<responseNames.size();++i){
                Response* r= getResponse(responseNames[i]);
                Tensor<StorageT>* features = new Tensor<StorageT>(r->dim);
                features->readGPU(r->dataGPU);

                size_t spel = numspel(r->dim);

                std::vector<int> receptive_field;       receptive_field.resize(r->receptive_field.size());
                std::vector<int> receptive_offset;      receptive_offset.resize(r->receptive_field.size());
                std::vector<int> receptive_gap;         receptive_gap.resize(r->receptive_field.size());

                for (int d=0;d<r->receptive_field.size();++d){
                    receptive_field [d] = r->receptive_field[d] /rData->receptive_field[d];
                    receptive_offset[d] = r->receptive_offset[d]/rData->receptive_field[d];
                    receptive_gap   [d] = r->receptive_gap[d]   /rData->receptive_field[d];
                }

                std::vector<int> dataDim;
                dataDim.push_back(dataChannels);
                dataDim.insert( dataDim.end(), receptive_field.begin(), receptive_field.end() );

                for (int j=0;j<responseChannels[i].size();++j){
                    int c = responseChannels[i][j];
                    if (c<0 || c >= r->dim[1]){
                        std::cerr<<"Channel exceeds maximal channel: Indexing Channel "<<c<<" outof "<<r->dim[1]<<" channels in "<<responseNames[i]<<std::endl;
                        FatalError(__LINE__);
                    }
                    for(int n=0; n<r->dim[0]; ++n){
                        for(size_t k=0; k<spel; ++k){
                            size_t idx = (n * r->dim[1] + c) * spel + k;
                            ComputeT val = CPUStorage2ComputeT(features->CPUmem[idx]);

                            if (scores[i][j].size()<topK || scoresLowest[i][j]< val){
                                Tensor<StorageT>* toSave = new Tensor<StorageT>(dataDim);

                                toSave->initialize(CPUCompute2StorageT(0));

                                // copy the n-th, all channesl, all spatal
                                if (dataDim.size()==3){
                                    // convert k to sx and sy
                                    int sx = receptive_offset[0] + k / features->dim[3] * receptive_gap[0];
                                    int sy = receptive_offset[1] + k % features->dim[3] * receptive_gap[1];

                                    for (int ic=0;ic<dataDim[0];++ic){
                                        for(int x=0; x<dataDim[1];++x){
                                            for(int y=0; y<dataDim[2];++y){
                                                if ( 0<=sx+x && sx+x< rData->dim[2] && 0<=sy+y && sy+y< rData->dim[3] ) {
                                                    size_t idxData  = ((n * rData->dim[1] + ic) * rData->dim[2] + (sx+x)) * rData->dim[3] + (sy+y);
                                                    size_t idxWrite = (ic * dataDim[1] + x) * dataDim[2] + y;
                                                    toSave->CPUmem[idxWrite] = rdataTensor->CPUmem[idxData];
                                                }
                                            }
                                        }
                                    }

                                }else if (dataDim.size()==4){
                                    // convert k to sx and sy
                                    int sx = receptive_offset[0] + k / (features->dim[3] * features->dim[4]) * receptive_gap[0];
                                    int sy = receptive_offset[1] + (k / features->dim[4]) % features->dim[3] * receptive_gap[1];
                                    int sz = receptive_offset[2] + k % features->dim[4] * receptive_gap[2];

                                    for (int ic=0;ic<dataDim[0];++ic){
                                        for(int x=0; x<dataDim[1];++x){
                                            for(int y=0; y<dataDim[2];++y){
                                                for(int z=0; z<dataDim[3];++z){
                                                    if ( 0<=sx+x && sx+x< rData->dim[2] && 0<=sy+y && sy+y< rData->dim[3] && 0<=sz+z && sz+z< rData->dim[4]) {
                                                        size_t idxData  = (((n * rData->dim[1] + ic) * rData->dim[2] + (sx+x)) * rData->dim[3] + (sy+y)) * rData->dim[4] + (sz+z);
                                                        size_t idxWrite = ((ic * dataDim[1] + x) * dataDim[2] + y) * dataDim[3] + z;
                                                        toSave->CPUmem[idxWrite] = rdataTensor->CPUmem[idxData];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }else{
                                    std::cerr<<"No implemented."<<std::endl;
                                    FatalError(__LINE__);
                                }

                                if (scores[i][j].size()<topK){
                                    scores[i][j].push_back(val);
                                    data[i][j].push_back(toSave);
                                    if (scores[i][j].size()==topK){
                                        std::vector<ComputeT>::iterator minEl = min_element(scores[i][j].begin(), scores[i][j].end());
                                        scoresLowest[i][j]= *minEl;
                                    }
                                }else{
                                    std::vector<ComputeT>::iterator minEl = min_element(scores[i][j].begin(), scores[i][j].end());
                                    int minID = std::distance(scores[i][j].begin(), minEl);
                                    scores[i][j][minID]=val;
                                    delete data[i][j][minID];
                                    data[i][j][minID] = toSave;
                                    minEl = min_element(scores[i][j].begin(), scores[i][j].end());
                                    scoresLowest[i][j]= *minEl;
                                }
                            }
                        }
                    }
                }
                delete features;
            }
            ++iter;
        }

        delete rdataTensor;

        // saving
        for(int i=0;i<responseNames.size();++i){
            for (int j=0;j<responseChannels[i].size();++j){
                std::vector<size_t> indices = sort_indexes(scores[i][j]);
                std::vector<Tensor<StorageT>*> toWrite;
                toWrite.resize(indices.size());
                std::cout<<responseNames[i]<<"_"<<responseChannels[i][j]<<": ";
                for(size_t k=0;k<indices.size();++k){
                    std::cout<<scores[i][j][indices[indices.size()-1-k]]<<" ";
                    toWrite[k]= data[i][j][indices[indices.size()-1-k]];
                }
                std::cout<<std::endl;

                std::string Fname = saveFilePrefix + responseNames[i] + "_" + std::to_string(responseChannels[i][j]) + ".tensor";
                while (is_file_exist(Fname)){
                    std::cerr<<"File "<<Fname<<" exists. Please delete it first. Will retry after 5 seconds."<<std::endl;
                    std::this_thread::sleep_for (std::chrono::seconds(5));
                }

                writeTensors<StorageT>(Fname, toWrite);
                for(size_t k=0;k<toWrite.size();++k){
                    delete toWrite[k];
                }
            }
        }
    };

    // for testing or extract feature, have to call after Malloc
    std::vector<ComputeT> test(std::vector<std::string> responseNames = std::vector<std::string>(), std::vector<std::string> saveFilenames = std::vector<std::string>(), int itersPerSave=0){

        phase = Testing;

        std::vector<ComputeT> result(loss_layers.size(), 0);
        std::vector<Tensor<StorageT>*> features(responseNames.size(),NULL);

        std::vector<FILE*> files(responseNames.size(),NULL);

        DataLayer* pDataLayer = NULL;
        for (int l=0; l<layers.size();++l){
            if (layers[l]->phase == phase || layers[l]->phase == TrainingTesting){
                if (layers[l]->isDataLayer()){
                    pDataLayer = (DataLayer*) layers[l];
                    break;
                }
            }
        }
        if (pDataLayer==NULL) { std::cerr<<"No data layer for Testing."<<std::endl; FatalError(__LINE__);};

        std::vector<size_t> total_size(responseNames.size());
        for(int i=0;i<responseNames.size();++i){
            Response* r = getResponse(responseNames[i]);
            std::vector<int> dim = r->dim;
            dim[0] = pDataLayer->numofitems();
            total_size[i] = numel(dim);
        }

        std::vector<int> file_counter(responseNames.size(),0);

        std::cout<< "====================================================================================================================================="<<std::endl;

        int iter = 0;

        while(pDataLayer->epoch == 0){
            resetLoss();
            forward();
            eval(false);

            // display
            if (display_iter >0 && iter % display_iter ==0) std::cout << "Iteration " << iter << "  ";
            for (int i=0;i<loss_layers.size();++i){
                if (loss_layers[i]->phase == phase || loss_layers[i]->phase == TrainingTesting){
                    if (display_iter >0 && iter % display_iter ==0) loss_layers[i]->display();
                    result[i] += loss_layers[i]->result;
                }
            }
            if (display_iter >0 && iter % display_iter ==0) std::cout << std::endl;

            // get features
            if (responseNames.size()>0){
                for(int i=0;i<responseNames.size();++i){
                    Response* r= getResponse(responseNames[i]);
                    if ((itersPerSave ==0 && iter==0) || (itersPerSave !=0 && iter % itersPerSave ==0)){

                        std::string Fname = saveFilenames[i];

                        if (itersPerSave!=0){
                            Fname = Fname + '_' + std::to_string(file_counter[i]) + ".tensor";
                        }

                        while (is_file_exist(Fname)){
                            std::cerr<<"File "<<Fname<<" exists. Please delete it first. Will retry after 5 seconds."<<std::endl;
                            std::this_thread::sleep_for (std::chrono::seconds(5));
                        }

                        Response* r = getResponse(responseNames[i]);
                        if (features[i]==NULL)
                            features[i] = new Tensor<StorageT>(r->dim);
                        files[i] = fopen(Fname.c_str(),"wb");

                        while (files[i]==NULL){
                            std::cerr<<"Open file "<<Fname<<" fails. Please check availablility of free disk space. Will retry after 5 seconds."<<std::endl;
                            std::this_thread::sleep_for(std::chrono::seconds(5));
                            files[i] = fopen(Fname.c_str(),"wb");
                        }

                        std::vector<int> dim = r->dim;
                        if (itersPerSave==0){
                            dim[0] = pDataLayer->numofitems();
                        }else{
                            int samplesPerFile = r->dim[0]*itersPerSave;
                            int samplesSaved = samplesPerFile * file_counter[i];

                            if (samplesSaved + samplesPerFile <= pDataLayer->numofitems()){
                                dim[0] = r->dim[0]*itersPerSave;
                            }else{
                                dim[0] = pDataLayer->numofitems() - samplesSaved;
                            }
                        }
                        features[i]->writeHeader(files[i],dim);

                        file_counter[i]++;
                    }

                    features[i]->readGPU(r-> dataGPU);
                    features[i]->writeData(files[i], total_size[i]);
                    total_size[i] -= features[i]->numel();

                    if (itersPerSave !=0 && iter % itersPerSave == itersPerSave-1){
                        fclose(files[i]);
                        files[i] = NULL;
                    }
                }
            }
            ++iter;
        }

        for(int i=0;i<responseNames.size();++i){
            if (files[i] != NULL){
                fclose(files[i]);
                files[i] = NULL;
            }
            delete(features[i]);
        }

        for (int i=0;i<result.size();++i){
            result[i] /= iter;
        }

        std::cout << "Average over " << iter << " iterations  ";
        for (int i=0;i<result.size();++i){
            if (loss_layers[i]->phase == phase || loss_layers[i]->phase == TrainingTesting){
                std::cout << " eval = " << result[i];
                std::cout << "  ";
            }
        }
        std::cout << std::endl;

        return result;
    };

};

//////////////////////////////////////////////////////////////////////////////////////////////////
// Solver
//////////////////////////////////////////////////////////////////////////////////////////////////

class Solver{
    bool singleGPU;
public:
    Phase phase;
    std::vector<Net* > nets;
    std::vector<std::thread> threads;

    std::string path;
    int iter;
    int current_step;
    int train_iter;

    std::vector<int> GPU;
    int GPU_solver;

    // machine learning paramters
    SolverAlgorithm solver;
    Regularizer regularizer;
    ComputeT momentum;
    ComputeT momentum2;
    ComputeT delta;
    ComputeT rms_decay;

    ComputeT weight_decay;
    ComputeT base_lr;
    LRPolicy lr_policy;
    ComputeT lr_gamma;
    ComputeT lr_power;
    int lr_stepsize;
    std::vector<int> stepvalue;
    int max_iter;           // maximum number of iterations
    int snapshot_iter;      // snapshot every N iterations
    int display_iter;       // Display every 100 iterations
    int test_iter;          // how many forward passes the test should carry out
    int test_interval;      // Carry out testing every 500 training iterations
    bool debug_mode;


    Solver(std::string filename=std::string()){

        // construct the network from the file in JSON
        JSON* train_obj = new JSON;
        JSON* architecture_obj = new JSON;
        parseNetworkJSON(filename, train_obj, NULL, architecture_obj);


        SetValue(train_obj, solver,         SGD)
        SetValue(train_obj, regularizer,    L2)
        SetValue(train_obj, momentum,       0.9)
        SetValue(train_obj, momentum2,      0.999)
        SetValue(train_obj, delta,          0.00000001)
        SetValue(train_obj, rms_decay,      0.98)

        SetValue(train_obj, weight_decay,   0.0005)
        SetValue(train_obj, base_lr,        0.01)
        SetValue(train_obj, lr_policy,      LR_inv)
        SetValue(train_obj, lr_gamma,       0.0001)
        SetValue(train_obj, lr_power,       0.75)
        SetValue(train_obj, lr_stepsize,    100000)
        SetValue(train_obj, train_iter,     1)
        SetValue(train_obj, max_iter,       10000)
        SetValue(train_obj, snapshot_iter,  5000)
        SetValue(train_obj, display_iter,   100)
        SetValue(train_obj, test_iter,      100)
        SetValue(train_obj, test_interval,  500)
        SetValue(train_obj, debug_mode,     false)
        SetValue(train_obj, GPU,            veci(1,0))
        SetOrDie(train_obj, path            )
        SetValue(train_obj, GPU_solver,     -1)

        if (GPU_solver==-1) GPU_solver=GPU[0];

        singleGPU = GPU.size()==1 && GPU_solver==GPU[0];

        int nGPUs = 0;
        checkCUDA(__LINE__, cudaGetDeviceCount(&nGPUs));
        if (nGPUs==0){
            std::cerr<<"There is no NVIDIA GPU available in this machine."<<std::endl;
            FatalError(__LINE__);
        }else if (nGPUs==1){
            std::cout<<"There is 1 NVIDIA GPU available in this machine."<<std::endl;
        }else{
            std::cout<<"There are "<< nGPUs<< " NVIDIA GPUs available in this machine."<<std::endl;
        }
        std::vector<int>::iterator largestGPUid = max_element(GPU.begin(), GPU.end());
        if (*largestGPUid>=nGPUs){
            std::cerr<<"Largest GPU ID request for GPU #"<<*largestGPUid<<" exceeds the number of available GPUs"<<std::endl;
            FatalError(__LINE__);
        }

        nets.resize(GPU.size());
        threads.resize(GPU.size());
        for (int n=0;n<nets.size();++n){
            nets[n] = new Net(architecture_obj, GPU[n]);
            nets[n]->debug_mode = debug_mode;
            nets[n]->train_iter = train_iter;
            nets[n]->test_iter  = test_iter;
        }

        delete train_obj;
        delete architecture_obj;

        if (GPU.size()>1){
            for (int n=0;n<GPU.size();++n){
                if (GPU[n]!=GPU_solver){

                    int canAccessPeer;
                    checkCUDA(__LINE__, cudaDeviceCanAccessPeer(&canAccessPeer, GPU[n], GPU_solver));

                    if (canAccessPeer==0){
                        std::cerr<<"Slave GPU #"<<GPU[n]<<" cannot access Master GPU #"<<GPU_solver<<std::endl;
                        FatalError(__LINE__);
                    }

                    checkCUDA(__LINE__, cudaSetDevice(GPU[n]));
                    checkCUDA(__LINE__, cudaDeviceEnablePeerAccess(GPU_solver, 0));
                }
            }
        }

    };

    ComputeT learning_rate(){
        ComputeT rate;
        switch(lr_policy){
            case LR_fixed:
                rate = base_lr;
                break;
            case LR_step:
                current_step = iter / lr_stepsize;
                rate = base_lr * pow(lr_gamma, current_step);
                break;
            case LR_exp:
                rate = base_lr * pow(lr_gamma, iter);
                break;
            case LR_inv:
                rate = base_lr * pow(ComputeT(1) + lr_gamma * iter, - lr_power);
                break;
            case LR_multistep:
                if (current_step < stepvalue.size() && iter >= stepvalue[current_step] ) {
                    current_step++;
                    std::cout << "MultiStep Status: Iteration " << iter << ", step = " << current_step << std::endl;
                }
                rate = base_lr * pow(lr_gamma, current_step);
                break;
            case LR_poly:
                rate = base_lr * pow(ComputeT(1) - (ComputeT(iter) / ComputeT(max_iter)), lr_power);
                break;
            case LR_sigmoid:
                rate = base_lr * (ComputeT(1) /  (ComputeT(1) + exp(-lr_gamma * (ComputeT(iter) - ComputeT(lr_stepsize)))));
                break;
            case LR_cyclical: // from http://arxiv.org/abs/1506.01186
                rate = 0; // TODO: place holder for now
                break;
        }
        return rate;
    };

    size_t Malloc(Phase phase_ = Training){
        phase = phase_;

        int nGPUs = 0;
        checkCUDA(__LINE__, cudaGetDeviceCount(&nGPUs));
        std::vector<size_t> memoryBytes(nGPUs,0);

        for (int n=0;n<nets.size();++n){
            memoryBytes[GPU[n]] += nets[n]->Malloc(phase);
        }

        size_t extraHistoryCount;

        switch (solver){
            case SGD:
                extraHistoryCount = 0;
                break;
            case AdaDelta:
                extraHistoryCount = 2;
                break;
            case AdaGrad:
                extraHistoryCount = 1;
                break;
            case Adam:
                extraHistoryCount = 2;
                break;
            case NAG:
                extraHistoryCount = 1;
                break;
            case RMSprop:
                extraHistoryCount = 1;
                break;
        }

        if (phase == Training || phase == TrainingTesting){
            checkCUDA(__LINE__,cudaSetDevice(GPU_solver));
            for (int l=0; l<nets[0]->layers.size(); ++l){
                if (nets[0]->layers[l]->train_me){
                    size_t weight_numel = nets[0]->layers[l]->weight_numel;
                    size_t   bias_numel = nets[0]->layers[l]->bias_numel;
                    if (weight_numel>0){
                        size_t weight_bytes = (1 + nets.size() + extraHistoryCount) * weight_numel * sizeofStorageT;
                        checkCUDA(__LINE__, cudaMalloc(&(nets[0]->layers[l]->weight_histGPU), weight_bytes));
                        checkCUDA(__LINE__, cudaMemset(nets[0]->layers[l]->weight_histGPU, 0, weight_bytes));
                        memoryBytes[GPU_solver] += weight_bytes;
                        for (int n=0;n<nets.size();++n){
                            nets[n]->layers[l]->weight_histGPU = nets[0]->layers[l]->weight_histGPU;
                            nets[n]->layers[l]->weight_diffGPU = nets[0]->layers[l]->weight_histGPU + weight_numel * (n+1);
                        }
                    }
                    if (bias_numel>0){
                        size_t bias_bytes = (1 + nets.size() + extraHistoryCount) * bias_numel * sizeofStorageT;
                        checkCUDA(__LINE__, cudaMalloc(&(nets[0]->layers[l]->bias_histGPU), bias_bytes));
                        checkCUDA(__LINE__, cudaMemset(nets[0]->layers[l]->bias_histGPU, 0, bias_bytes));
                        memoryBytes[GPU_solver] += bias_bytes;
                        for (int n=0;n<nets.size();++n){
                            nets[n]->layers[l]->bias_histGPU = nets[0]->layers[l]->bias_histGPU;
                            nets[n]->layers[l]->bias_diffGPU = nets[0]->layers[l]->bias_histGPU + bias_numel * (n+1);
                        }
                    }
                    
                    for (int ll=0;ll<nets[0]->layers[l]->sub_layers.size(); ++ll){
                        if (nets[0]->layers[l]->sub_layers[ll]->train_me){
                            size_t weight_numel = nets[0]->layers[l]->sub_layers[ll]->weight_numel;
                            size_t   bias_numel = nets[0]->layers[l]->sub_layers[ll]->bias_numel;
                            if (weight_numel>0){
                                size_t weight_bytes = (1 + nets.size() + extraHistoryCount) * weight_numel * sizeofStorageT;
                                checkCUDA(__LINE__, cudaMalloc(&(nets[0]->layers[l]->sub_layers[ll]->weight_histGPU), weight_bytes));
                                checkCUDA(__LINE__, cudaMemset(nets[0]->layers[l]->sub_layers[ll]->weight_histGPU, 0, weight_bytes));
                                memoryBytes[GPU_solver] += weight_bytes;
                                for (int n=0;n<nets.size();++n){
                                    nets[n]->layers[l]->sub_layers[ll]->weight_histGPU = nets[0]->layers[l]->sub_layers[ll]->weight_histGPU;
                                    nets[n]->layers[l]->sub_layers[ll]->weight_diffGPU = nets[0]->layers[l]->sub_layers[ll]->weight_histGPU + weight_numel * (n+1);
                                }
                            }
                            if (bias_numel>0){
                                size_t bias_bytes = (1 + nets.size() + extraHistoryCount) * bias_numel * sizeofStorageT;
                                checkCUDA(__LINE__, cudaMalloc(&(nets[0]->layers[l]->sub_layers[ll]->bias_histGPU), bias_bytes));
                                checkCUDA(__LINE__, cudaMemset(nets[0]->layers[l]->sub_layers[ll]->bias_histGPU, 0, bias_bytes));
                                memoryBytes[GPU_solver] += bias_bytes;
                                for (int n=0;n<nets.size();++n){
                                    nets[n]->layers[l]->sub_layers[ll]->bias_histGPU = nets[0]->layers[l]->sub_layers[ll]->bias_histGPU;
                                    nets[n]->layers[l]->sub_layers[ll]->bias_diffGPU = nets[0]->layers[l]->sub_layers[ll]->bias_histGPU + bias_numel * (n+1);
                                }
                            }
                        }

                    }
                }
            }
        }

        std::cout<< "====================================================================================================================================="<<std::endl;
        for (int n=0;n<nGPUs;++n){
            if (memoryBytes[n]>0){
                std::cout<< "GPU " << n << ": Total GPU memory: ";  memorySizePrint(memoryBytes[n]);    std::cout<<std::endl;
            }
        }

        size_t totalMemory = memoryBytes[0];
        for (int n=1;n<nGPUs;++n){
            totalMemory += memoryBytes[n];
        }

        std::cout<< "All GPUs: Total GPU memory: "; memorySizePrint(totalMemory);   std::cout<<std::endl;

        return totalMemory;
    };

    ~Solver(){
        checkCUDA(__LINE__,cudaSetDevice(GPU_solver));
        for (int l=0; l<nets[0]->layers.size(); ++l){
            if (nets[0]->layers[l]->train_me){
                if (nets[0]->layers[l]->weight_numel>0){
                    if (nets[0]->layers[l]->weight_histGPU!=NULL) checkCUDA(__LINE__, cudaFree(nets[0]->layers[l]->weight_histGPU));
                }
                if (nets[0]->layers[l]->bias_numel>0){
                    if (nets[0]->layers[l]->bias_histGPU!=NULL) checkCUDA(__LINE__, cudaFree(nets[0]->layers[l]->bias_histGPU));
                }

                for(int ll=0; ll<nets[0]->layers[l]->sub_layers.size(); ++ll){
                    if (nets[0]->layers[l]->sub_layers[ll]->train_me){
                        if (nets[0]->layers[l]->sub_layers[ll]->weight_numel>0){
                            if (nets[0]->layers[l]->sub_layers[ll]->weight_histGPU!=NULL) checkCUDA(__LINE__, cudaFree(nets[0]->layers[l]->sub_layers[ll]->weight_histGPU));
                        }
                        if (nets[0]->layers[l]->sub_layers[ll]->bias_numel>0){
                            if (nets[0]->layers[l]->sub_layers[ll]->bias_histGPU!=NULL) checkCUDA(__LINE__, cudaFree(nets[0]->layers[l]->sub_layers[ll]->bias_histGPU));
                        }
                    }
                }
            }
        }
    };

    void randInit(){
        nets[0]->randInit();
        for (int n=1;n<nets.size();++n){
            for (int l=0; l<nets[0]->layers.size(); ++l){
                if (nets[0]->layers[l]->weight_numel>0) checkCUDA(__LINE__, cudaMemcpy(nets[n]->layers[l]->weight_dataGPU, nets[0]->layers[l]->weight_dataGPU, nets[0]->layers[l]->weight_numel*sizeofStorageT, cudaMemcpyDeviceToDevice) );
                if (nets[0]->layers[l]->bias_numel>0)   checkCUDA(__LINE__, cudaMemcpy(nets[n]->layers[l]->bias_dataGPU, nets[0]->layers[l]->bias_dataGPU, nets[0]->layers[l]->bias_numel*sizeofStorageT, cudaMemcpyDeviceToDevice) );
                for(int ll=0; ll<nets[0]->layers[l]->sub_layers.size(); ++ll){
                    if (nets[0]->layers[l]->sub_layers[ll]->weight_numel>0) checkCUDA(__LINE__, cudaMemcpy(nets[n]->layers[l]->sub_layers[ll]->weight_dataGPU, nets[0]->layers[l]->sub_layers[ll]->weight_dataGPU, nets[0]->layers[l]->sub_layers[ll]->weight_numel*sizeofStorageT, cudaMemcpyDeviceToDevice) );
                    if (nets[0]->layers[l]->sub_layers[ll]->bias_numel>0)   checkCUDA(__LINE__, cudaMemcpy(nets[n]->layers[l]->sub_layers[ll]->bias_dataGPU, nets[0]->layers[l]->sub_layers[ll]->bias_dataGPU, nets[0]->layers[l]->sub_layers[ll]->bias_numel*sizeofStorageT, cudaMemcpyDeviceToDevice) );
                }
            }
        }
    };

    void solve(ComputeT learning_rate){
        checkCUDA(__LINE__,cudaSetDevice(GPU_solver));
        for (int l=0; l<nets[0]->layers.size(); ++l){
            if (nets[0]->layers[l]->train_me){
                if (nets[0]->layers[l]->weight_numel>0)
                    update_solver(solver, regularizer, iter, nets[0]->layers[l]->weight_numel, nets.size(), weight_decay * nets[0]->layers[l]->weight_decay_mult, momentum, momentum2, delta, rms_decay, learning_rate * nets[0]->layers[l]->weight_lr_mult, nets[0]->layers[l]->weight_dataGPU, nets[0]->layers[l]->weight_histGPU);
                if (nets[0]->layers[l]->bias_numel>0)
                    update_solver(solver, regularizer, iter, nets[0]->layers[l]->bias_numel, nets.size(), weight_decay * nets[0]->layers[l]->bias_decay_mult, momentum, momentum2, delta, rms_decay, learning_rate * nets[0]->layers[l]->bias_lr_mult, nets[0]->layers[l]->bias_dataGPU, nets[0]->layers[l]->bias_histGPU);

                for(int ll=0; ll<nets[0]->layers[l]->sub_layers.size(); ++ll){
                    if (nets[0]->layers[l]->sub_layers[ll]->train_me){
                        if (nets[0]->layers[l]->sub_layers[ll]->weight_numel>0)
                            update_solver(solver, regularizer, iter, nets[0]->layers[l]->sub_layers[ll]->weight_numel, nets.size(), weight_decay * nets[0]->layers[l]->sub_layers[ll]->weight_decay_mult, momentum, momentum2, delta, rms_decay, learning_rate * nets[0]->layers[l]->sub_layers[ll]->weight_lr_mult, nets[0]->layers[l]->sub_layers[ll]->weight_dataGPU, nets[0]->layers[l]->sub_layers[ll]->weight_histGPU);
                        if (nets[0]->layers[l]->sub_layers[ll]->bias_numel>0)
                            update_solver(solver, regularizer, iter, nets[0]->layers[l]->sub_layers[ll]->bias_numel, nets.size(), weight_decay * nets[0]->layers[l]->sub_layers[ll]->bias_decay_mult, momentum, momentum2, delta, rms_decay, learning_rate * nets[0]->layers[l]->sub_layers[ll]->bias_lr_mult, nets[0]->layers[l]->sub_layers[ll]->bias_dataGPU, nets[0]->layers[l]->sub_layers[ll]->bias_histGPU);
                    }
                }

            }
        }
    };

    void loadWeights(std::string filename, bool diff=false){

        std::cout<< "====================================================================================================================================="<<std::endl;

        std::vector<Tensor<StorageT>*> weights = readTensors<StorageT>(filename);

        for (int i=0;i<nets.size();++i){
            nets[i]->loadWeights(weights, diff);
        }

        for (int i=0; i<weights.size();++i){
            delete weights[i];
        }
    };

    void saveWeights(std::string filename, bool diff=false){
        nets[0]->saveWeights(filename, diff);
    };

    void train(int iter_begin = 0){

        checkCUDA(__LINE__,cudaSetDevice(GPU_solver));

        phase = Training;
        current_step = 0;

        std::cout<< "====================================================================================================================================="<<std::endl;
        std::cout<< "  Training:                                                                      Testing:                                            "<<std::endl;
        std::cout<< "====================================================================================================================================="<<std::endl;

        for (iter=iter_begin;iter<=max_iter;++iter){

            if (iter % test_interval==0 && test_iter > 0){
                if (debug_mode){
                    std::cout << "Testing Iteration " << iter << std::endl;
                }else{
                    std::cout<< "                                                                                 ";
                    std::cout << "Iteration " << iter;
                }

                if (singleGPU){
                    nets[0]->phase = Testing;
                    nets[0]->stepTest(false);
                    nets[0]->phase = Training;
                }else{
                    for (int t=0; t<threads.size(); ++t){
                        nets[t]->phase = Testing;
                        threads[t] = std::thread(&Net::stepTest, nets[t], true);    //nets[t]->stepTest();
                    }
                    for (int t=0; t<threads.size(); ++t){
                        threads[t].join();
                        nets[t]->phase = Training;
                    }
                }

                for (int l=0; l<nets[0]->loss_layers.size();++l){
                    if (nets[0]->loss_layers[l]->phase == phase || nets[0]->loss_layers[l]->phase == TrainingTesting){
                        for (int t=1;t<nets.size(); ++t){
                            nets[0]->loss_layers[l]->result += nets[t]->loss_layers[l]->result;
                            nets[0]->loss_layers[l]->loss += nets[t]->loss_layers[l]->loss;
                        }
                        nets[0]->loss_layers[l]->result /= nets.size();
                        nets[0]->loss_layers[l]->loss   /= nets.size();
                        nets[0]->loss_layers[l]->display();
                    }
                }
                std::cout << std::endl;
            }

            if (singleGPU){
                nets[0]->stepTrain(false);
            }else{
                for (int t=0; t<threads.size(); ++t){
                    threads[t] = std::thread(&Net::stepTrain, nets[t], true);   //nets[t]->stepTrain();
                }
                for (int t=0; t<threads.size(); ++t){
                    threads[t].join();
                }
            }

            ComputeT lrate = learning_rate();
            solve(lrate);
            checkCUDA(__LINE__,cudaDeviceSynchronize());

            if (iter!=iter_begin && iter % snapshot_iter==0){
                saveWeights(path+"_snapshot_"+std::to_string(iter)+".marvin",false);
            }
            if (iter % display_iter==0){
                std::cout << "Iteration " << iter << "  ";
                std::cout << "learning_rate = "<< lrate;


                if (singleGPU){
                    nets[0]->eval(false);
                }else{
                    for (int t=0; t<threads.size(); ++t){
                        threads[t] = std::thread(&Net::eval, nets[t], true); //nets[t]->eval();
                    }
                    for (int t=0; t<threads.size(); ++t){
                        threads[t].join();
                    }
                }

                for (int l=0; l<nets[0]->loss_layers.size();++l){
                    if (nets[0]->loss_layers[l]->phase == phase || nets[0]->loss_layers[l]->phase == TrainingTesting){
                        for (int t=1;t<threads.size(); ++t){
                            nets[0]->loss_layers[l]->result += nets[t]->loss_layers[l]->result;
                            nets[0]->loss_layers[l]->loss += nets[t]->loss_layers[l]->loss;
                        }
                        nets[0]->loss_layers[l]->result /= threads.size();
                        nets[0]->loss_layers[l]->loss   /= threads.size();
                        nets[0]->loss_layers[l]->display();
                    }
                }
                std::cout << std::endl;
            }
        }
    };
};

}  // namespace marvin

#endif