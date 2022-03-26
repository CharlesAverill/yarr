/**
 * @file
 * @author Charles Averill
 * @date   04-Feb-2022
 * @brief Description
*/

#include <SFML/Graphics.h>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "canvas.cuh"
#include "input.cuh"
#include "linear_algebra/vector.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/utils.cuh"

#define BLOCK_SIZE 16

sfRenderWindow *csfml_setup(unsigned int width, unsigned int height)
{
    sfVideoMode mode = {width, height, 32};

    sfRenderWindow *window = sfRenderWindow_create(mode, "YARR", sfResize | sfClose, NULL);
    if (!window) {
        fprintf(stderr, "[csfml_setup] Couldn't initialize SFML window");
        exit(1);
    }

    sfRenderWindow_setFramerateLimit(window, 60);

    return window;
}

void render_loop(Canvas *canvas, sfRenderWindow *window, const char *output_fn)
{
    int frame_number = 0;

    sfEvent event;
    sfTexture *texture;
    sfSprite *sprite;

    cv::Mat frame(canvas->width, canvas->height, CV_8UC4, cv::Scalar(0, 0, 0));
    cv::VideoWriter oVideoWriter(output_fn,
                                 cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                 TARGET_FPS,
                                 cv::Size(canvas->width, canvas->height),
                                 true);
    if (!oVideoWriter.isOpened()) {
        fprintf(stderr, "[render_loop] Failed to initialize video writer\n");
        exit(1);
    }

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
        canvas->update(frame_number++);

        // Render output
        canvas->render();

        // Append frame to output video
        for (int x = 0; x < canvas->width; x++) {
            for (int y = 0; y < canvas->height; y++) {
                int index = canvas->channels * (x * canvas->width + y);

                frame.data[index + 0] = canvas->canvas[index + 0];
                frame.data[index + 1] = canvas->canvas[index + 1];
                frame.data[index + 2] = canvas->canvas[index + 2];
                frame.data[index + 3] = canvas->canvas[index + 3];
            }
        }
        oVideoWriter.write(frame);

        sfRenderWindow_clear(window, sfBlack);

        sfTexture_updateFromPixels(texture, canvas->canvas, canvas->width, canvas->height, 0, 0);
        sfRenderWindow_drawSprite(window, sprite, NULL);

        sfRenderWindow_display(window);
    }

    oVideoWriter.release();
    sfSprite_destroy(sprite);
    sfTexture_destroy(texture);
    sfRenderWindow_destroy(window);
}

int main(int argc, char *argv[])
{
    // Deal with input arguments
    const char *output_fn;
    if (argc < 2) {
        output_fn = "yarr.avi";
    } else {
        output_fn = argv[1];
    }

    // These are the dimensions of a 4D (RGBA) matrix that we will flatten into 1D
    int width = 512;
    int height = 512;
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
    canvas->host_setup(grid_size, block_size);

    // Setup window
    sfRenderWindow *window = csfml_setup(width, height);

    // Call render loop here
    render_loop(canvas, window, output_fn);

    // Free memory
    cudaFree(canvas);
}
