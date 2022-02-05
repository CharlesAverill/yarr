/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef CANVAS_H
#define CANVAS_H

#include <stdio.h>

typedef struct canvas {
	int width;
    int height;
    int channels;

    int *values;
} canvas;

int canvas_to_ppm(canvas *c, char *fn);

int canvas_size(canvas *c);

#endif
