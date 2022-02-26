/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

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

#endif
