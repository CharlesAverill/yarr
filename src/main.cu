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

    // Get device information from CUDA
    int device_ID;
    cudaDeviceProp props;

    cudaGetDevice(&device_ID);
    cudaGetDeviceProperties(&props, device_ID);

    // Calculate our kernel dimensions
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_size(int(ceil(float(width) / float(BLOCK_SIZE))), int(ceil(float(height) / float(BLOCK_SIZE))), channels);

    // Instantiate our Canvas object
    Canvas *canvas;
    cudaMallocManaged(&canvas, sizeof(Canvas));
    canvas->init(width, height, channels);

    // Our array of color values (0 - 255) of shape [R, G, B]
    int *color;
    cudaMallocManaged(&color, sizeof(int) * 3);
    hex_str_to_color_arr(color, "FF00FF");

    // Initialize our canvas on the GPU
    canvas->render(grid_size, block_size, color);

    // Synchronize and check for errors
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());

    // Save canvas to PPM
    printf("Saving render to %s\n", output_fn);
    canvas->save_to_ppm(output_fn);

    // Free memory
    cudaFree(canvas);
}
