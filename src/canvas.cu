/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "canvas.cuh"

int Canvas::get_size() {
    return this->width * this->height * this->channels;
}

int Canvas::save_to_ppm(char *fn) {
    // Open file
    FILE *fp;
    fp = fopen(fn, "w+");

    if(fp == NULL) {
        return 1;
    }

    // Store canvas size
    int size = this->get_size();

    // Write header
    fprintf(fp, "P3 %d %d 255 ", this->width, this->height);

    for(int i = 0; i < size; i++) {
        fprintf(fp, "%d ", this->canvas[i]);
    }

    fclose(fp);

    return 0;
}

__global__ void render_kernel(int *values, int width, int height, int channels, Vector<int> *color) {
    // Kernel row and column based on their thread and block indices
    int x = (threadIdx.x + blockIdx.x * blockDim.x) - (width / 2);
    int y = (threadIdx.y + blockIdx.y * blockDim.y) - (height / 2);
    int color_index = threadIdx.z + blockIdx.z * blockDim.z;
    // The 1D index of `canvas` given our 3D information
    int index = ((x + (width / 2)) * height * channels) + ((y + (height / 2)) * channels) + color_index;

    if (x >= width || y >= height || index >= width * height * channels) {
        return;
    }

    int new_value;
    switch(color_index) {
        case 0:
            new_value = color->x;
            break;
        case 1:
            new_value = color->y;
            break;
        case 2:
            new_value = color->z;
            break;
    }
    values[index] = new_value; //color[color_index];
}

void Canvas::render(dim3 grid_size, dim3 block_size, Vector<int> *color) {
    render_kernel<<<grid_size, block_size>>>(this->canvas, this->width, this->height, this->channels, color);
}
