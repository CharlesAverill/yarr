/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CANVAS_H
#define CANVAS_H

#include <SFML/Graphics.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>

#include <stdio.h>
#include <time.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <curand.h>
#include <curand_kernel.h>

#include "light/point.cuh"
#include "renderobjects/octahedron.cuh"
#include "renderobjects/sphere.cuh"
#include "renderobjects/triangle.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/list.cuh"
#include "utils/settings.cuh"
#include "utils/utils.cuh"
#include "utils/vector.cuh"

class Canvas
{
  public:
    // Dimensions of our image
    int width;
    int height;
    int channels;
    int size;

    // Kernel dimensions
    dim3 block_size;
    dim3 grid_size;

    // Array containing our image RGB values
    sfUint8 *canvas;

    // Coordinate Vectors
    Vector<float> *X;
    Vector<float> *Y;
    Vector<float> *Z;

    // Viewport
    Vector<float> *viewport_origin;

    // Scene RenderObjects
    int num_renderobjects;
    RenderObject **scene_renderobjects;

    // Scene Lights
    int num_lights;
    Point **scene_lights;

    // Ground texture
    int ground_texture_width;
    int ground_texture_height;
    int ground_texture_channels;
    sfUint8 *ground_texture;

    // Antialiasing Colors
    int antialiasing_samples;
    Vector<int> **antialiasing_colors_array;

    // Constructors
    Canvas(int w, int h, int c)
    {
        init(w, h, c);
    }

    void init(int w, int h, int c)
    {
        width = w;
        height = h;
        channels = c;
        size = w * h * c;

        cudaMallocManaged(&canvas, width * height * channels);

        cudaMallocManaged(&X, sizeof(Vector<float>));
        cudaMallocManaged(&Y, sizeof(Vector<float>));
        cudaMallocManaged(&Z, sizeof(Vector<float>));

        cudaMallocManaged(&viewport_origin, sizeof(Vector<float>));

        X->init(0.002f, 0, 0);
        Y->init(0, 0.002f, 0);
        Z->init(0, 0, 1);

        viewport_origin->init(0, 1, -4);

        // Ground texture setup
        if (GROUND_TYPE == GT_TEXTURE) {
            cv::Mat tempTexture = cv::imread(GROUND_TEXTURE_PATH, cv::IMREAD_COLOR);
            if (tempTexture.data == NULL) {
                fprintf(stderr,
                        "[canvas::init] Error reading ground texture \"%s\"\n",
                        GROUND_TEXTURE_PATH);
                exit(1);
            }

            // Read from OpenCV struct to color array
            ground_texture_width = tempTexture.size().width;
            ground_texture_height = tempTexture.size().height;
            ground_texture_channels = tempTexture.channels();
            cudaMallocManaged(&ground_texture,
                              ground_texture_width * ground_texture_height *
                                  ground_texture_channels);

            for (int i = 0; i < ground_texture_width; i++) {
                for (int j = 0; j < ground_texture_height; j++) {
                    int index = ground_texture_channels * (i * ground_texture_width + j);

                    ground_texture[index + 0] = tempTexture.data[index + 0];
                    ground_texture[index + 1] = tempTexture.data[index + 1];
                    ground_texture[index + 2] = tempTexture.data[index + 2];
                    ground_texture[index + 3] = tempTexture.data[index + 3];
                }
            }
        }
    }

    // Save canvas to PPM file
    int save_to_ppm(const char *fn);
    int save_to_ppm(const char *fn, sfUint8 *to_save, int width, int height, int channels);

    void scene_setup();
    void host_setup(dim3 grid_size, dim3 block_size);

    // Run render pipeline on GPU
    void render();

    // Convert an integer to a vector of colors
    __hd__ void hex_int_to_color_vec(Vector<int> *out, int in);

    // Getters
    __hd__ Vector<float> *get_X()
    {
        return X;
    }
    __hd__ Vector<float> *get_Y()
    {
        return Y;
    }
    __hd__ Vector<float> *get_Z()
    {
        return Z;
    }

    // Setters
    __host__ void set_kernel_size(dim3 grid_size, dim3 block_size)
    {
        this->grid_size = grid_size;
        this->block_size = block_size;
    }
};

#endif
