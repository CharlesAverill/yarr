/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "canvas.cuh"

int Canvas::save_to_ppm(char *fn)
{
    // Open file
    FILE *fp;
    fp = fopen(fn, "w+");

    if (fp == NULL) {
        return 1;
    }

    // Store canvas size
    int size = this->size;

    // Write header
    fprintf(fp, "P3 %d %d 255 ", this->width, this->height);

    for (int i = 0; i < size; i++) {
        fprintf(fp, "%d ", (int)this->canvas[i]);
    }

    fclose(fp);

    return 0;
}

__hd__ void Canvas::hex_int_to_color_vec(Vector<int> *out, int in)
{
    long mask1 = 255;
    long mask2 = 65280;
    long mask3 = 16711680;

    out->init((in & mask3) >> 16, (in & mask2) >> 8, in & mask1);
}

__device__ void get_sky_color(Vector<int> *color, Vector<float> ray, Canvas *canvas)
{
    canvas->hex_int_to_color_vec(color, 0xB399FF);
    (*color) = (*color) * pow(1 - ray.y, 2);
}

__device__ void
get_ground_color(Vector<int> *color, Vector<float> *ray_origin, Vector<float> ray, Canvas *canvas)
{
    float distance = -1 * ray_origin->y / ray.y;
    float x        = ray_origin->x + distance * ray.x;
    float z        = ray_origin->z + distance * ray.z;

    if ((int)abs(floor(x)) % 2 == (int)abs(floor(z)) % 2) {
        canvas->hex_int_to_color_vec(color, 0xFF0000);
    } else {
        canvas->hex_int_to_color_vec(color, 0xFFFFFF);
    }
}

__global__ void render_kernel(Canvas *canvas)
{
    // Kernel row and column based on their thread and block indices
    int x           = (threadIdx.x + blockIdx.x * blockDim.x) - (canvas->width / 2);
    int y           = (threadIdx.y + blockIdx.y * blockDim.y) - (canvas->height / 2);
    int color_index = threadIdx.z + blockIdx.z * blockDim.z;
    // The 1D index of `canvas` given our 3D information
    int index = ((y + (canvas->width / 2)) * canvas->height * canvas->channels) +
                ((x + (canvas->height / 2)) * canvas->channels) + color_index;

    // Bounds checking
    if (x >= canvas->width || y >= canvas->height || index >= canvas->size) {
        return;
    }

    // Create color vector
    Vector<int> out_color;

    // Initialize the ray
    Vector<float> ray_direction = (*(canvas->get_X()) * float(x)) +
                                  (*(canvas->get_Y()) * float(y) * -1) + (*(canvas->get_Z()));
    ray_direction            = !ray_direction;
    Vector<float> ray_origin = *(canvas->viewport_origin);

    // Cast the ray
    float hit_distance;
    Vector<float> ray_collide_position;
    Vector<float> ray_reflect_direction;
    float hit_reflectiveness;
    float ray_energy = 1.f;

    for (int reflectionIndex = 0; reflectionIndex <= MAX_REFLECTIONS; reflectionIndex++) {
        Vector<int> bounce_color;

        // Check for intersection with each triangle
        bool hit_object        = false;
        float min_hit_distance = C_INFINITY;
        Triangle *closest_triangle;

        for (int triangle_index = 0; triangle_index < canvas->num_triangles; triangle_index++) {
            Triangle *test_hit = &(canvas->scene_triangles)[triangle_index];
            if (is_visible(test_hit,
                           ray_origin,
                           ray_direction,
                           ray_collide_position,
                           ray_reflect_direction,
                           hit_distance,
                           bounce_color,
                           hit_reflectiveness)) {
                hit_object = true;
                if (hit_distance < min_hit_distance) {
                    min_hit_distance = hit_distance;
                    closest_triangle = test_hit;
                }
            }
        }

        // Check for sky or ground plane
        if (hit_object && closest_triangle) {
            is_visible(closest_triangle,
                       ray_origin,
                       ray_direction,
                       ray_collide_position,
                       ray_reflect_direction,
                       hit_distance,
                       bounce_color,
                       hit_reflectiveness);
            ray_origin    = ray_collide_position;
            ray_direction = ray_reflect_direction;
        } else {
            if (ray_direction.y < 0) {
                get_ground_color(&bounce_color, &ray_origin, ray_direction, canvas);
                hit_reflectiveness = 0.f;
            } else {
                get_sky_color(&bounce_color, ray_direction, canvas);
                hit_reflectiveness = 0.f;
            }
        }

        // Update color and ray energy
        out_color = out_color + (bounce_color * (ray_energy * (1 - hit_reflectiveness)));
        ray_energy *= hit_reflectiveness;

        if(ray_energy <= 0.f) {
            break;
        }
    }

    // Save color data
    canvas->canvas[index]     = out_color.x;
    canvas->canvas[index + 1] = out_color.y;
    canvas->canvas[index + 2] = out_color.z;
}

void Canvas::render(dim3 grid_size, dim3 block_size)
{
    // Initialize triangles
    thrust::host_vector<Triangle> host_triangles;

    Triangle *blue_triangle;
    cudaMallocManaged(&blue_triangle, sizeof(Triangle));
    init_triangle(blue_triangle,
                  Vector<float>(-2, 0, -1),
                  Vector<float>(2, 0, -1),
                  Vector<float>(0, 3, -1),
                  Vector<int>(0, 0, 255),
                  0.8f);
    host_triangles.push_back(*blue_triangle);

    Triangle *green_triangle;
    cudaMallocManaged(&green_triangle, sizeof(Triangle));
    init_triangle(green_triangle,
                  Vector<float>(2, 0, -5),
                  Vector<float>(-2, 0, -5),
                  Vector<float>(0, 3, -5),
                  Vector<int>(0, 255, 0),
                  0.8f);
    host_triangles.push_back(*green_triangle);

    // Copy triangles to device
    cudaMallocManaged(&(this->scene_triangles), sizeof(Triangle) * host_triangles.size());
    cudaMemcpy(this->scene_triangles,
               thrust::raw_pointer_cast(host_triangles.data()),
               sizeof(Triangle) * host_triangles.size(),
               cudaMemcpyHostToDevice);
    this->num_triangles = host_triangles.size();

    // Run render kernel
    render_kernel<<<grid_size, block_size>>>(this);
}
