/**
 * @file
 * @author Charles Averill
 * @date   04-Feb-2022
 * @brief Description
*/

#include <SFML/Graphics.h>
#include <math.h>
#include <stdio.h>

#include "canvas.cuh"
#include "cuda_utils.cuh"
#include "input.cuh"
#include "utils.cuh"
#include "vector.cuh"

#define BLOCK_SIZE 16

sfRenderWindow *csfml_setup(int width, int height)
{
    sfVideoMode mode = {width, height, 32};

    sfRenderWindow *window = sfRenderWindow_create(mode, "YARR", sfResize | sfClose, NULL);
    if (!window) {
        printf("Couldn't initialize SFML window");
        exit(1);
    }

    sfRenderWindow_setFramerateLimit(window, 60);

    return window;
}

void render_loop(Canvas *canvas, sfRenderWindow *window)
{
    sfEvent event;
    sfTexture *texture;
    sfTexture *texture2;
    sfSprite *sprite;

    texture = sfTexture_create(canvas->width, canvas->height);
    if (!texture) {
        return;
    }

    sprite = sfSprite_create();
    sfSprite_setTexture(sprite, texture, sfTrue);

    while (sfRenderWindow_isOpen(window)) {
        // Process Input
        input_loop(window, &event);

        // Update Scene

        // Render output
        canvas->render();

        gpuErrorCheck(cudaPeekAtLastError());
        gpuErrorCheck(cudaDeviceSynchronize());

        sfRenderWindow_clear(window, sfBlack);

        sfTexture_updateFromPixels(texture, canvas->canvas, canvas->width, canvas->height, 0, 0);
        sfRenderWindow_drawSprite(window, sprite, NULL);

        sfRenderWindow_display(window);
    }

    sfSprite_destroy(sprite);
    sfTexture_destroy(texture);
    sfRenderWindow_destroy(window);
}

int main(int argc, char *argv[])
{
    // Deal with input arguments
    char *output_fn;
    if (argc < 2) {
        output_fn = "yarr.ppm";
    } else {
        output_fn = argv[1];
    }

    // These are the dimensions of a 4D (RGBA) matrix that we will flatten into 1D
    int width    = 1024;
    int height   = 1024;
    int channels = 4;

    // Get device information from CUDA
    int device_ID;
    cudaDeviceProp props;

    cudaGetDevice(&device_ID);
    cudaGetDeviceProperties(&props, device_ID);

    // Calculate our kernel dimensions
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid_size(int(ceil(float(width) / float(BLOCK_SIZE))),
                   int(ceil(float(height) / float(BLOCK_SIZE))),
                   1);

    // Instantiate our Canvas object
    Canvas *canvas;
    cudaMallocManaged(&canvas, sizeof(Canvas));
    canvas->init(width, height, channels);

    // Setup scene
    canvas->scene_setup();
    canvas->set_kernel_size(grid_size, block_size);

    // Setup window
    sfRenderWindow *window = csfml_setup(width, height);

    // Call render loop here
    render_loop(canvas, window);

    // Save last render to PPM
    printf("Saving render to %s\n", output_fn);
    canvas->save_to_ppm(output_fn);

    // Free memory
    cudaFree(canvas);
}
