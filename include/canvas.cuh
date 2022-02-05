/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CANVAS_H
#define CANVAS_H

#ifdef __CUDACC__
#define __hd__ __host__ __device__
#else
#define __hd__
#endif

#include <stdio.h>

class Canvas {
public:
	int width;
	int height;
	int channels;

	int *canvas;

	Canvas(int w, int h, int c) {
		init(w, h, c);
	}

	void init(int w, int h, int c) {
		width = w;
		height = h;
		channels = c;

		cudaMallocManaged(&canvas, width * height * channels * 4);
	}

	int get_size();
	int save_to_ppm(char *fn);

	void render(dim3 grid_size, dim3 block_size, int color[3]);
};

#endif
