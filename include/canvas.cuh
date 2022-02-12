/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CANVAS_H
#define CANVAS_H

#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "cuda_utils.cuh"
#include "triangle.cuh"
#include "utils.cuh"
#include "vector.cuh"

class Canvas {
public:
	// Dimensions of our image
	int width;
	int height;
	int channels;
	int size;

	// Array containing our image RGB values
	int *canvas;

	// Coordinate Vectors
	Vector<float> *X;
	Vector<float> *Y;
	Vector<float> *Z;

	// Viewport
	Vector<float> *viewport_origin;

	// Scene Triangles
	int num_triangles;
	Triangle *scene_triangles;

	// Constructors
	Canvas(int w, int h, int c) {
		init(w, h, c);
	}

	void init(int w, int h, int c) {
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

		viewport_origin->init(0, 1, 0);
	}

	// Save canvas to PPM file
	int save_to_ppm(char *fn);

	// Run render pipeline on GPU
	void render(dim3 grid_size, dim3 block_size);

	// Convert an integer to a vector of colors
	__hd__ void hex_int_to_color_vec(Vector<int> *out, int in);

	// Getters
	__hd__ Vector<float> *get_X() { return X; }
	__hd__ Vector<float> *get_Y() { return Y; }
	__hd__ Vector<float> *get_Z() { return Z; }
};

#endif
