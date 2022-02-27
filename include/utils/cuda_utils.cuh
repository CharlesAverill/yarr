/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <curand.h>
#include <curand_kernel.h>

#include <math.h>
#include <stdio.h>

#ifdef __CUDACC__
#define __hd__ __host__ __device__
#else
#define __hd__
#endif

#define gpuErrorCheck(ans)                                                                         \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// cuRAND API errors
static const char *curandGetErrorString(curandStatus_t error)
{
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";

    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define curandErrorCheck(x)                                                                        \
    {                                                                                              \
        curandassert((x), __FILE__, __LINE__);                                                     \
    }

inline void curandassert(curandStatus_t code, const char *file, int line, bool abort = true)
{
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRANDassert: %s %s %d\n", curandGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#endif
