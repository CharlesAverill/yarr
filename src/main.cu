/**
 * @file
 * @author Charles Averill
 * @date   04-Feb-2022
 * @brief Description
*/

#include <stdio.h>
#include <math.h>

#include "utils.cuh"
#include "canvas.cuh"

#define BLOCK_SIZE 16

__global__ void init_canvas(canvas *c, int c_size, int color[3]) {
    // Kernel row and column based on their thread and block indices
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z + blockIdx.z * blockDim.z;
    // The 1D index of `canvas` given our 3D information
    // This represents a pixel, so we will still loop 3 times to fill out RGB information
    int index = (row * c->height * c->channels) + (col * c->channels) + z;

    if (row >= c->width || col >= c->height || index >= c_size) {
        return;
    }

    c->values[index] = color[z];
}

int main(int argc, char *argv[]) {
    // Deal with input arguments
    char *output_fn;
    if(argc < 2) {
        output_fn = "yarr.ppm";
    } else {
        output_fn = argv[1];
    }

    // These are the dimensions of a 3D matrix that we will flatten into 1D
    int width = 512;
    int height = 512;
    int channels = 3;

    // Our array of color values (0 - 255) of shape [R, G, B]
    int *color;
    cudaMallocManaged(&color, sizeof(int) * 3);
    hex_str_to_color_arr(color, "FF00FF");

    // Initialize our canvas struct
    canvas *c;
    cudaMallocManaged(&c, sizeof(canvas));
    // I'm not yet sure why I need to multiply the size by 4 here, but without it I run into
    // GPUassert: an illegal memory access was encountered
    cudaMallocManaged(&(c->values), width * height * channels * 4);
    c->width = width;
    c->height = height;
    c->channels = channels;

    // Get device information from CUDA
    int device_ID;
    cudaDeviceProp props;

    cudaGetDevice(&device_ID);
    cudaGetDeviceProperties(&props, device_ID);

    // Calculate our kernel dimensions
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_size(int(ceil(float(width) / float(BLOCK_SIZE))), int(ceil(float(height) / float(BLOCK_SIZE))), channels);

    // Initialize our canvas on the GPU
    init_canvas<<<grid_size, block_size>>>(c, canvas_size(c), color);

    // Synchronize and check for errors
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    // Save canvas to PPM
    printf("Saving render to %s\n", output_fn);
    canvas_to_ppm(c, output_fn);

    // Free memory
    cudaFree(c);
}
