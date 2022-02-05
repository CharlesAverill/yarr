/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CANVAS_H
#define CANVAS_H

#include <stdio.h>

#include "cuda_utils.cuh"
#include "vector.cuh"

class Canvas {
private:
	// Dimensions of our image
	int width;
	int height;
	int channels;

	// Array containing our image RGB values
	int *canvas;
public:
	// Constructors
	Canvas(int w, int h, int c) {
		init(w, h, c);
	}

	void init(int w, int h, int c) {
		width = w;
		height = h;
		channels = c;

		cudaMallocManaged(&canvas, width * height * channels * 4);
	}

	// Return size of canvas
	int get_size();
	// Save canvas to PPM file
	int save_to_ppm(char *fn);

	// Run render pipeline on GPU
	void render(dim3 grid_size, dim3 block_size, Vector<int> *color);
};

#endif
