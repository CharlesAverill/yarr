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
        fprintf(fp, "%d ", (int)this->canvas[i]);
    }

    fclose(fp);

    return 0;
}

__hd__ void Canvas::hex_int_to_color_vec(Vector<int> *out, int in) {
    long mask1 = 255;
    long mask2 = 65280;
    long mask3 = 16711680;

    out->init((in & mask3) >> 16,
              (in & mask2) >> 8,
              in & mask1);
}

__device__ void get_sky_color(Vector<int> *color, Vector<float> ray, Canvas *canvas) {
    canvas->hex_int_to_color_vec(color, 0xB399FF);
    (*color) = (*color) * pow(1 - ray.y, 2);
}

__device__ void get_ground_color(Vector<int> *color, Vector<float> *ray_origin, Vector<float> ray, Canvas *canvas) {
    float distance = -1 * ray_origin->y / ray.y;
    float x = ray_origin->x + distance * ray.x;
    float z = ray_origin->z + distance * ray.z;

    if((int)abs(floor(x)) % 2 == (int)abs(floor(z)) % 2) {
        canvas->hex_int_to_color_vec(color, 0xFF0000);
    } else {
        canvas->hex_int_to_color_vec(color, 0xFFFFFF);
    }
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
    Vector<float> ray_direction = (*(canvas->get_X()) * float(x)) + (*(canvas->get_Y()) * float(y) * -1) + (*(canvas->get_Z()));
    ray_direction = !ray_direction;

    // Check for intersection with each triangle
    float hit_distance;
    bool hit_object = false;
    float min_hit_distance = C_INFINITY;

    for(int i = 0; i < canvas->num_triangles; i++) {
        Triangle *test_hit = &(canvas->scene_triangles)[i];
        Vector<int> test_color;
        if(is_visible(test_hit, *(canvas->viewport_origin), ray_direction, hit_distance, test_color)) {
            hit_object = true;
            if(hit_distance < min_hit_distance) {
                min_hit_distance = hit_distance;
                color = test_color;
            }
        }
    }

    // Check for sky or ground plane
    if(!hit_object) {
        if(ray_direction.y < 0) {
            get_ground_color(&color, canvas->viewport_origin, ray_direction, canvas);
        } else {
            get_sky_color(&color, ray_direction, canvas);
        }
    }

    // Save color data
    canvas->canvas[index] = color.x;
    canvas->canvas[index + 1] = color.y;
    canvas->canvas[index + 2] = color.z;
}

void Canvas::render(dim3 grid_size, dim3 block_size) {
    thrust::host_vector<Triangle> host_triangles;

    Triangle *blue_triangle;
    cudaMallocManaged(&blue_triangle, sizeof(Triangle));
    init_triangle(blue_triangle, Vector<float>(-1, 0, 0), Vector<float>(1, 0, 0), Vector<float>(0, 1.73, 0), Vector<int>(0, 0, 255));
    host_triangles.push_back(*blue_triangle);

    Triangle *green_triangle;
    cudaMallocManaged(&green_triangle, sizeof(Triangle));
    init_triangle(green_triangle, Vector<float>(2, 0, 2), Vector<float>(1, 1.73, 2), Vector<float>(0, 0, 2), Vector<int>(0, 255, 0));
    host_triangles.push_back(*green_triangle);

    Triangle *red_triangle;
    cudaMallocManaged(&red_triangle, sizeof(Triangle));
    init_triangle(red_triangle, Vector<float>(-0.25, 0.75, -1), Vector<float>(0.75, 0.75, -1), Vector<float>(0.25, 2, -1), Vector<int>(255, 0, 0));
    host_triangles.push_back(*red_triangle);

    cudaMallocManaged(&(this->scene_triangles), sizeof(Triangle) * host_triangles.size());
    cudaMemcpy(this->scene_triangles, thrust::raw_pointer_cast(host_triangles.data()), sizeof(Triangle) * host_triangles.size(), cudaMemcpyHostToDevice);
    this->num_triangles = host_triangles.size();

    render_kernel<<<grid_size, block_size>>>(this);
}
