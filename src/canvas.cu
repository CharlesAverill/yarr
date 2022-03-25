/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "canvas.cuh"

int Canvas::save_to_ppm(const char *fn)
{
    // Open file
    FILE *fp = fopen(fn, "w+");
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

int Canvas::save_to_ppm(const char *fn, sfUint8 *to_save, int width, int height, int channels)
{
    // Open file
    FILE *fp = fopen(fn, "w+");
    if (fp == NULL) {
        return 1;
    }

    // Store canvas size
    int size = width * height * channels;

    // Write header
    fprintf(fp, "P3 %d %d 255 ", width, height);

    for (int i = 0; i < size; i++) {
        // Skip the Alpha channel
        if (i > 0 && (i + 1) % 4 == 0) {
            continue;
        }

        fprintf(fp, "%d ", (int)to_save[i]);
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

    switch (GROUND_TYPE) {
    case GT_PLAIN:
        canvas->hex_int_to_color_vec(color, GROUND_PRIMARY_COLOR);
        break;
    case GT_CHECKER:
        if ((int)abs(floor(x)) % 2 == (int)abs(floor(z)) % 2) {
            canvas->hex_int_to_color_vec(color, GROUND_PRIMARY_COLOR);
        } else {
            canvas->hex_int_to_color_vec(color, GROUND_SECONDARY_COLOR);
        }
        break;
    case GT_HSTRIPE:
        if ((int)abs(floor(z)) % 2 == 0) {
            canvas->hex_int_to_color_vec(color, GROUND_PRIMARY_COLOR);
        } else {
            canvas->hex_int_to_color_vec(color, GROUND_SECONDARY_COLOR);
        }
        break;
    case GT_VSTRIPE:
        if ((int)abs(floor(x)) % 2 == 0) {
            canvas->hex_int_to_color_vec(color, GROUND_PRIMARY_COLOR);
        } else {
            canvas->hex_int_to_color_vec(color, GROUND_SECONDARY_COLOR);
        }
        break;
    case GT_RINGS:
        if ((int)sqrt(pow(abs(x), 2) + pow(abs(z), 2)) % 2 == 0) {
            canvas->hex_int_to_color_vec(color, GROUND_PRIMARY_COLOR);
        } else {
            canvas->hex_int_to_color_vec(color, GROUND_SECONDARY_COLOR);
        }
        break;
    case GT_TEXTURE:
        int texture_x = abs((int)(x * 100) + 1000) % canvas->ground_texture_width;
        int texture_z = abs((int)(z * 100) + 1000) % canvas->ground_texture_height;

        int color_start_idx = (texture_x * canvas->ground_texture_width + texture_z) *
                              canvas->ground_texture_channels;

        color->init(canvas->ground_texture[color_start_idx + 0],
                    canvas->ground_texture[color_start_idx + 1],
                    canvas->ground_texture[color_start_idx + 2]);
        break;
    }
}

/*
__global__ void antialiasing_kernel(Canvas *canvas, int x, int y, curandState randState)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= canvas->antialiasing_samples) {
        return;
    }

    Vector<int> bounce_color;
    Vector<int> out_color;

    // Initialize the ray
    Vector<float> ray_origin = *(canvas->viewport_origin);
    Vector<float> ray_direction =
        (*(canvas->get_X()) * float(x + (DO_ANTIALIASING) * (curand_uniform(&randState) - 1))) +
        (*(canvas->get_Y()) * float(y + (DO_ANTIALIASING) * (curand_uniform(&randState) - 1)) *
         -1) +
        (*(canvas->get_Z()));
    ray_direction = !ray_direction;

    // Depth of field
    Vector<float> sensor_shift{
        (curand_uniform(&randState) - 0.5f) * DOF, (curand_uniform(&randState) - 0.5f) * DOF, 0};
    ray_origin = ray_origin + sensor_shift;
    ray_direction = !(ray_direction - sensor_shift * 0.25f);

    // Cast the ray
    Vector<float> ray_collide_position;
    Vector<float> ray_reflect_direction;
    Vector<float> hit_normal;
    float hit_distance;

    float hit_metallic;
    float hit_hardness;
    float hit_diffuse;
    float hit_specular;

    float ray_energy = 1.f;

    for (int reflectionIndex = 0; reflectionIndex <= MAX_REFLECTIONS; reflectionIndex++) {
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
                                     hit_specular,

                                     Vector<float>(curand_uniform(&randState) - 0.5f,
                                                   curand_uniform(&randState) - 0.5f,
                                                   curand_uniform(&randState) - 0.5f))) {
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
                                             hit_specular,

                                             Vector<float>(curand_uniform(&randState) - 0.5f,
                                                           curand_uniform(&randState) - 0.5f,
                                                           curand_uniform(&randState) - 0.5f));

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

                hit_metallic = GROUND_METALLIC;
                hit_hardness = GROUND_HARDNESS;
                hit_diffuse = GROUND_DIFFUSE;
                hit_specular = GROUND_SPECULAR;
            } else {
                get_sky_color(&bounce_color, ray_direction, canvas);

                hit_sky = true;
                hit_metallic = 0.f;
                hit_hardness = 0.f;
                hit_specular = 0.f;
            }
        }

        bool point_directly_hit = true;
        for (int light_index = 0; light_index < canvas->num_lights; light_index++) {
            Point *current_light = canvas->scene_lights[light_index];
            Vector<float> light_position =
                current_light->position +
                Vector<float>((curand_uniform(&randState) - 0.5f) * SOFT_SHADOW_FACTOR,
                              0,
                              (curand_uniform(&randState) - 0.5f) * SOFT_SHADOW_FACTOR);
            for (int renderobject_index = 0; renderobject_index < canvas->num_renderobjects;
                 renderobject_index++) {
                Vector<float> _dummy_fvector;
                Vector<int> _dummy_ivector;
                float _dummy_float;

                if (canvas->scene_renderobjects[renderobject_index]->is_visible(
                        ray_collide_position,
                        !(light_position - ray_collide_position),
                        _dummy_fvector,
                        _dummy_fvector,
                        _dummy_fvector,
                        _dummy_float,
                        _dummy_ivector,
                        _dummy_float,
                        _dummy_float,
                        _dummy_float,
                        _dummy_float,
                        Vector<float>(0, 0, 0))) {
                    point_directly_hit = false;
                    break;
                }
            }

            // Compute lighting
            if (!hit_sky) {
                if (point_directly_hit) {
                    // Compute sum of diffuse and specular for all lights in scene
                    float diffuse = max(0.f, hit_normal % !(light_position - ray_collide_position));
                    float specular =
                        max(0.f, !(light_position - ray_collide_position) % ray_direction);

                    // Phong lighting
                    bounce_color =
                        (bounce_color * AMBIENT_LIGHT) + (bounce_color * diffuse * hit_diffuse) +
                        (current_light->color * pow(specular, hit_hardness) * hit_specular);
                } else {
                    bounce_color = bounce_color * AMBIENT_LIGHT;
                }
            }
        }

        // Update color and ray energy
        out_color = out_color + (bounce_color * (ray_energy * (1 - hit_metallic)));
        ray_energy *= hit_metallic;

        if (ray_energy <= 0.f) {
            break;
        }
    }

    cudaMemcpyAsync(canvas->antialiasing_colors_array[index],
                    &out_color,
                    sizeof(Vector<int>),
                    cudaMemcpyDeviceToDevice);
}
*/

__global__ void render_kernel(Canvas *canvas)
{
    // Kernel row and column based on their thread and block indices
    int x = (threadIdx.x + blockIdx.x * blockDim.x) - (canvas->height / 2);
    int y = (threadIdx.y + blockIdx.y * blockDim.y) - (canvas->width / 2);
    int color_index = threadIdx.z + blockIdx.z * blockDim.z;
    // The 1D index of `canvas` given our 3D information
    int index = ((y + (canvas->width / 2)) * canvas->width * canvas->channels) +
                ((x + (canvas->height / 2)) * canvas->channels) + color_index;

    // Init RNG
    curandState randState;
    // Should be the same for every run
    curand_init(SEED, index, 0, &randState);

    // Bounds checking
    if (x >= canvas->width || y >= canvas->height || index >= canvas->size) {
        return;
    }

    Vector<int> antialiasing_color;

    for (int antialiasing_index = 0; antialiasing_index < canvas->antialiasing_samples;
         antialiasing_index++) {
        Vector<int> bounce_color;
        Vector<int> out_color;

        // Initialize the ray
        Vector<float> ray_origin = *(canvas->viewport_origin);
        Vector<float> ray_direction =
            (*(canvas->get_X()) * float(x + (DO_ANTIALIASING) * (curand_uniform(&randState) - 1))) +
            (*(canvas->get_Y()) * float(y + (DO_ANTIALIASING) * (curand_uniform(&randState) - 1)) *
             -1) +
            (*(canvas->get_Z()));
        ray_direction = !ray_direction;

        // Depth of field
        Vector<float> sensor_shift{(curand_uniform(&randState) - 0.5f) * DOF,
                                   (curand_uniform(&randState) - 0.5f) * DOF,
                                   0};
        ray_origin = ray_origin + sensor_shift;
        ray_direction = !(ray_direction - sensor_shift * 0.25f);

        // Cast the ray
        Vector<float> ray_collide_position;
        Vector<float> ray_reflect_direction;
        Vector<float> hit_normal;
        float hit_distance;

        float hit_metallic;
        float hit_hardness;
        float hit_diffuse;
        float hit_specular;

        float ray_energy = 1.f;

        for (int reflectionIndex = 0; reflectionIndex <= MAX_REFLECTIONS; reflectionIndex++) {
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
                                         hit_specular,

                                         Vector<float>(curand_uniform(&randState) - 0.5f,
                                                       curand_uniform(&randState) - 0.5f,
                                                       curand_uniform(&randState) - 0.5f))) {
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
                                                 hit_specular,

                                                 Vector<float>(curand_uniform(&randState) - 0.5f,
                                                               curand_uniform(&randState) - 0.5f,
                                                               curand_uniform(&randState) - 0.5f));

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

                    hit_metallic = GROUND_METALLIC;
                    hit_hardness = GROUND_HARDNESS;
                    hit_diffuse = GROUND_DIFFUSE;
                    hit_specular = GROUND_SPECULAR;
                } else {
                    get_sky_color(&bounce_color, ray_direction, canvas);

                    hit_sky = true;
                    hit_metallic = 0.f;
                    hit_hardness = 0.f;
                    hit_specular = 0.f;
                }
            }

            bool point_directly_hit = true;
            for (int light_index = 0; light_index < canvas->num_lights; light_index++) {
                Point *current_light = canvas->scene_lights[light_index];
                Vector<float> light_position =
                    current_light->position +
                    Vector<float>((curand_uniform(&randState) - 0.5f) * SOFT_SHADOW_FACTOR,
                                  0,
                                  (curand_uniform(&randState) - 0.5f) * SOFT_SHADOW_FACTOR);
                for (int renderobject_index = 0; renderobject_index < canvas->num_renderobjects;
                     renderobject_index++) {
                    Vector<float> _dummy_fvector;
                    Vector<int> _dummy_ivector;
                    float _dummy_float;

                    if (canvas->scene_renderobjects[renderobject_index]->is_visible(
                            ray_collide_position,
                            !(light_position - ray_collide_position),
                            _dummy_fvector,
                            _dummy_fvector,
                            _dummy_fvector,
                            _dummy_float,
                            _dummy_ivector,
                            _dummy_float,
                            _dummy_float,
                            _dummy_float,
                            _dummy_float,
                            Vector<float>(0, 0, 0))) {
                        point_directly_hit = false;
                        break;
                    }
                }

                // Compute lighting
                if (!hit_sky) {
                    if (point_directly_hit) {
                        // Compute sum of diffuse and specular for all lights in scene
                        float diffuse =
                            max(0.f, hit_normal % !(light_position - ray_collide_position));
                        float specular =
                            max(0.f, !(light_position - ray_collide_position) % ray_direction);

                        // Phong lighting
                        bounce_color =
                            (bounce_color * AMBIENT_LIGHT) +
                            (bounce_color * diffuse * hit_diffuse) +
                            (current_light->color * pow(specular, hit_hardness) * hit_specular);
                    } else {
                        bounce_color = bounce_color * AMBIENT_LIGHT;
                    }
                }
            }

            // Update color and ray energy
            out_color = out_color + (bounce_color * (ray_energy * (1 - hit_metallic)));
            ray_energy *= hit_metallic;

            if (ray_energy <= 0.f) {
                break;
            }
        }

        antialiasing_color = antialiasing_color + out_color;
    }

    antialiasing_color = antialiasing_color * (1.f / canvas->antialiasing_samples);

    // Save color data
    canvas->canvas[index] = max(0, min(255, antialiasing_color.x));
    canvas->canvas[index + 1] = max(0, min(255, antialiasing_color.y));
    canvas->canvas[index + 2] = max(0, min(255, antialiasing_color.z));
    // Alpha
    canvas->canvas[index + 3] = 255;
}

__global__ void scene_setup_kernel(Canvas *canvas)
{
    // Initialize RenderObjects
    List<RenderObject *> renderobjects;

    // Octahedron
    Octahedron *oct = new Octahedron(Vector<float>{0, 1, 0}, 1);
    oct->extend_list(&renderobjects);

    // Spheres
    renderobjects.add(
        new Sphere(Vector<float>{1, 2, 0}, 0.5f, Vector<int>{0, 0, 255}, 0.4f, 99, 0.9f, 1, 0));
    renderobjects.add(new Sphere(Vector<float>{-0.75, 0.35, -0.5},
                                 0.25f,
                                 Vector<int>{255, 165, 0},
                                 0.05f,
                                 99,
                                 0.9f,
                                 1,
                                 0.5));

    // Copy RenderObjects to canvas
    canvas->scene_renderobjects =
        (RenderObject **)malloc(sizeof(RenderObject *) * renderobjects.size());
    if (canvas->scene_renderobjects == NULL) {
        printf("[scene_setup_kernel] Failed to allocate memory for scene RenderObjects\n");
        return;
    }
    cudaMemcpyAsync(canvas->scene_renderobjects,
                    renderobjects.getArray(),
                    sizeof(RenderObject *) * renderobjects.size(),
                    cudaMemcpyDeviceToDevice);
    canvas->num_renderobjects = renderobjects.size();

    // Initialize Lights
    List<Point *> lights;
    lights.add(new Point(Vector<float>(0, 100, 0), Vector<int>(255, 255, 255)));

    // Copy Lights to canvas
    canvas->scene_lights = (Point **)malloc(sizeof(Point *) * lights.size());
    if (canvas->scene_lights == NULL) {
        printf("[scene_setup_kernel] Failed to allocate memory for scene Lights\n");
        return;
    }
    cudaMemcpyAsync(canvas->scene_lights,
                    lights.getArray(),
                    sizeof(Point *) * lights.size(),
                    cudaMemcpyDeviceToDevice);
    canvas->num_lights = lights.size();

    // Allocate memory for antialiasing
    canvas->antialiasing_samples = DO_ANTIALIASING ? ANTIALIASING_SAMPLES : 1;
    canvas->antialiasing_colors_array = (Vector<int> **)malloc(
        sizeof(Vector<int> *) * canvas->antialiasing_samples * canvas->width * canvas->height);
    if (canvas->antialiasing_colors_array == NULL) {
        printf("[scene_setup_kernel] Failed to allocate memory for antialiasing array\n");
        return;
    }
    for (int i = 0; i < canvas->antialiasing_samples; i++) {
        canvas->antialiasing_colors_array[i] = (Vector<int> *)malloc(sizeof(Vector<int>));
    }
}

void Canvas::scene_setup()
{
    scene_setup_kernel<<<1, 1>>>(this);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());
}

void Canvas::host_setup(dim3 grid_size, dim3 block_size)
{
    // Set heap size to ~67mb
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 67108864);
    scene_setup();
    set_kernel_size(grid_size, block_size);
}

void Canvas::render()
{
    // Run render kernel
    render_kernel<<<this->grid_size, this->block_size>>>(this);
    gpuErrorCheck(cudaPeekAtLastError());
    gpuErrorCheck(cudaDeviceSynchronize());
}
