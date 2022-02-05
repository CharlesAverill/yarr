/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

#include "canvas.cuh"

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void arr_to_ppm(canvas *c, char *fn);

void hex_str_to_color_arr(int out[3], char in[6]);

#endif
