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
        // Skip the Alpha channel
        if (i > 0 && (i + 1) % 4 == 0) {
            continue;
        }

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

__device__ void get_ground_color(Vector<int> *color,
                                 Vector<float> *ray_origin,
                                 Vector<float> ray,
                                 Vector<float> &ray_collide_position,
                                 Canvas *canvas)
{
    float distance = -1 * ray_origin->y / ray.y;
    float x = ray_origin->x + distance * ray.x;
    float z = ray_origin->z + distance * ray.z;

    ray_collide_position = Vector<float>(x, 0, z);

    if ((int)abs(floor(x)) % 2 == (int)abs(floor(z)) % 2) {
        canvas->hex_int_to_color_vec(color, 0xFF0000);
    } else {
        canvas->hex_int_to_color_vec(color, 0xFFFFFF);
    }
}

__global__ void render_kernel(Canvas *canvas)
{
    // Kernel row and column based on their thread and block indices
    int x = (threadIdx.x + blockIdx.x * blockDim.x) - (canvas->width / 2);
    int y = (threadIdx.y + blockIdx.y * blockDim.y) - (canvas->height / 2);
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
    ray_direction = !ray_direction;
    Vector<float> ray_origin = *(canvas->viewport_origin);

    // Cast the ray
    float hit_distance;
    Vector<float> ray_collide_position;
    Vector<float> ray_reflect_direction;
    Vector<float> hit_normal;

    float hit_metallic;
    float hit_hardness;
    float hit_diffuse;
    float hit_specular;

    float ray_energy = 1.f;

    for (int reflectionIndex = 0; reflectionIndex <= MAX_REFLECTIONS; reflectionIndex++) {
        Vector<int> bounce_color;

        bool hit_object = false;
        bool hit_sky = false;
        float min_hit_distance = C_INFINITY;

        RenderObject *closest_renderobject = NULL;

        // Check for intersection with each RenderObject
        for (int index = 0; index < canvas->num_renderobjects; index++) {
            RenderObject *test_hit = canvas->scene_renderobjects[index];
            if (test_hit->is_visible(ray_origin,
                                     ray_direction,
                                     ray_collide_position,
                                     ray_reflect_direction,
                                     hit_normal,
                                     hit_distance,

                                     bounce_color,
                                     hit_metallic,
                                     hit_hardness,
                                     hit_diffuse,
                                     hit_specular)) {
                hit_object = true;
                if (hit_distance < min_hit_distance) {
                    min_hit_distance = hit_distance;
                    closest_renderobject = test_hit;
                }
            }
        }

        // Check for sky or ground plane
        if (hit_object && closest_renderobject) {
            closest_renderobject->is_visible(ray_origin,
                                             ray_direction,
                                             ray_collide_position,
                                             ray_reflect_direction,
                                             hit_normal,
                                             hit_distance,

                                             bounce_color,
                                             hit_metallic,
                                             hit_hardness,
                                             hit_diffuse,
                                             hit_specular);

            ray_origin = ray_collide_position;
            ray_direction = ray_reflect_direction;
        } else {
            if (ray_direction.y < 0) {
                get_ground_color(
                    &bounce_color, &ray_origin, ray_direction, ray_collide_position, canvas);
                hit_normal = Vector<float>{0, 1, 0};
                ray_reflect_direction =
                    !(ray_direction + !hit_normal * (-ray_direction % hit_normal) * 2);

                ray_origin = ray_collide_position;
                ray_direction = ray_reflect_direction;

                hit_metallic = 0.f;
                hit_hardness = 0.f;
                hit_diffuse = GROUND_DIFFUSE;
                hit_specular = 0.f;
            } else {
                get_sky_color(&bounce_color, ray_direction, canvas);

                hit_sky = true;
                hit_metallic = 0.f;
                hit_hardness = 0.f;
                hit_specular = 0.f;
            }
        }

        // Compute lighting
        if (!hit_sky) {
            // Compute sum of diffuse and specular for all lights in scene
            float diffuse_sum = 0;
            float specular_sum = 0;
            Vector<int> light_color_sum = Vector<int>(0, 0, 0);

            float light_color_average_divisor = (1.f / (float)canvas->num_lights);
            for (int light_index = 0; light_index < canvas->num_lights; light_index++) {
                Point *current_light = canvas->scene_lights[light_index];

                diffuse_sum += current_light->diffuse_at_point(hit_normal, ray_collide_position);
                specular_sum +=
                    current_light->specular_at_point(ray_collide_position, ray_direction);

                light_color_sum.x += int(current_light->color.x * light_color_average_divisor);
                light_color_sum.y += int(current_light->color.y * light_color_average_divisor);
                light_color_sum.z += int(current_light->color.z * light_color_average_divisor);
            }

            // Phong lighting
            bounce_color = (bounce_color * AMBIENT_LIGHT) +
                           (bounce_color * diffuse_sum * hit_diffuse) +
                           (light_color_sum * pow(specular_sum, hit_hardness) * hit_specular);
        }

        // Update color and ray energy
        out_color = out_color + (bounce_color * (ray_energy * (1 - hit_metallic)));
        ray_energy *= hit_metallic;

        if (ray_energy <= 0.f) {
            break;
        }
    }

    // Save color data
    canvas->canvas[index] = max(0, min(255, out_color.x));
    canvas->canvas[index + 1] = max(0, min(255, out_color.y));
    canvas->canvas[index + 2] = max(0, min(255, out_color.z));
    // Alpha
    canvas->canvas[index + 3] = 255;
}

__global__ void scene_setup_kernel(Canvas *canvas)
{
    // Initialize RenderObjects
    List<RenderObject *> renderobjects;

    // Octahedron
    Octahedron *oct = new Octahedron(Vector<float>{0, 1, 0}, 1.0f);
    oct->extend_list(&renderobjects);

    // Spheres
    renderobjects.add(
        new Sphere(Vector<float>{1, 2, 0}, 0.5f, Vector<int>{0, 0, 255}, 0.25f, 99, 0.9f, 1, 0));
    renderobjects.add(new Sphere(
        Vector<float>{-0.75, 0.2, 0}, 0.25f, Vector<int>{255, 165, 0}, 0.05f, 99, 0.9f, 1, 0));

    // Copy RenderObjects to canvas
    canvas->scene_renderobjects =
        (RenderObject **)malloc(sizeof(RenderObject *) * renderobjects.size());
    cudaMemcpyAsync(canvas->scene_renderobjects,
                    renderobjects.getArray(),
                    sizeof(RenderObject *) * renderobjects.size(),
                    cudaMemcpyDeviceToDevice);
    canvas->num_renderobjects = renderobjects.size();

    // Initialize Lights
    List<Point *> lights;
    lights.add(new Point(Vector<float>(0, 2.5, 0), Vector<int>(255, 255, 255)));

    // Copy Lights to canvas
    canvas->scene_lights = (Point **)malloc(sizeof(Point *) * lights.size());
    cudaMemcpyAsync(canvas->scene_lights,
                    lights.getArray(),
                    sizeof(Point *) * lights.size(),
                    cudaMemcpyDeviceToDevice);
    canvas->num_lights = lights.size();
}

void Canvas::scene_setup()
{
    scene_setup_kernel<<<1, 1>>>(this);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());
}

void Canvas::render()
{
    // Run render kernel
    render_kernel<<<this->grid_size, this->block_size>>>(this);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());
}
