/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "canvas.cuh"

int canvas_size(canvas *c) {
    return c->width * c->height * c->channels;
}

int canvas_to_ppm(canvas *c, char *fn) {
    // Open file
    FILE *fp;
    fp = fopen(fn, "w+");

    if(fp == NULL) {
        return 1;
    }

    // Store canvas size
    int size = canvas_size(c);

    // Write header
    fprintf(fp, "P3 %d %d 255 ", c->width, c->height);

    for(int i = 0; i < size; i++) {
        fprintf(fp, "%d ", c->values[i]);
    }

    fclose(fp);

    return 0;
}
