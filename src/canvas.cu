/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "canvas.cuh"

int Canvas::save_to_ppm(char *fn) {
    // Open file
    FILE *fp;
    fp = fopen(fn, "w+");

    if(fp == NULL) {
        return 1;
    }

    // Store canvas size
    int size = this->size;

    // Write header
    fprintf(fp, "P3 %d %d 255 ", this->width, this->height);

    for(int i = 0; i < size; i++) {
        fprintf(fp, "%d ", this->canvas[i]);
    }

    fclose(fp);

    return 0;
}

__hd__ void Canvas::hex_int_to_color_vec(Vector<int> *out, int in) {
    long mask1 = 255;
    long mask2 = 65280;
    long mask3 = 16711680;

    out->init(int((in & mask3) >> 16),
              int((in & mask2) >> 8),
              int(in & mask1));
}

__device__ void get_sky_color(Vector<int> *color, Vector<float> ray, Canvas *canvas) {
    canvas->hex_int_to_color_vec(color, 0xFF0000);
}

__device__ void get_ground_color(Vector<int> *color, Vector<float> *ray_origin, Vector<float> ray, Canvas *canvas) {
    canvas->hex_int_to_color_vec(color, 0x0000FF);
}

__global__ void render_kernel(Canvas *canvas) {
    // Kernel row and column based on their thread and block indices
    int x = (threadIdx.x + blockIdx.x * blockDim.x) - (canvas->width / 2);
    int y = (threadIdx.y + blockIdx.y * blockDim.y) - (canvas->height / 2);
    int color_index = threadIdx.z + blockIdx.z * blockDim.z;
    // The 1D index of `canvas` given our 3D information
    int index = ((y + (canvas->width / 2)) * canvas->height * canvas->channels) +
                 ((x + (canvas->height / 2)) * canvas->channels) +
                 color_index;

    // Bounds checking
    if (x >= canvas->width || y >= canvas->height || index >= canvas->size) {
        return;
    }

    // Create color vector
    Vector<int> color;

    // Raycast to determine pixel color
    Vector<float> ray_direction = (*(canvas->get_X()) * x) + (*(canvas->get_Y()) * y) + (*(canvas->get_Z()));

    if(ray_direction.y < 0) {
        get_ground_color(&color, canvas->viewport_origin, ray_direction, canvas);
    } else {
        get_sky_color(&color, ray_direction, canvas);
    }

    // Save color data
    canvas->canvas[index] = color.x;
    canvas->canvas[index + 1] = color.y;
    canvas->canvas[index + 2] = color.z;
}

void Canvas::render(dim3 grid_size, dim3 block_size) {
    render_kernel<<<grid_size, block_size>>>(this);
}
