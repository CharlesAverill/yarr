/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CANVAS_H
#define CANVAS_H

#include <SFML/Graphics.h>
#include <stdio.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

    // Scene Triangles
    int num_renderobjects;
    RenderObject **scene_renderobjects;

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

        cudaMallocManaged(&canvas, width * height * channels * 4);

        cudaMallocManaged(&X, sizeof(Vector<float>));
        cudaMallocManaged(&Y, sizeof(Vector<float>));
        cudaMallocManaged(&Z, sizeof(Vector<float>));

        cudaMallocManaged(&viewport_origin, sizeof(Vector<float>));

        X->init(0.002f, 0, 0);
        Y->init(0, 0.002f, 0);
        Z->init(0, 0, 1);

        viewport_origin->init(0, 1, -4);
    }

    // Save canvas to PPM file
    int save_to_ppm(char *fn);

    void scene_setup();

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
