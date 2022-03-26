/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   26-Feb-2022
 * @brief Description
*/

#ifndef OCTAHEDRON_H
#define OCTAHEDRON_H

#include "renderobjects/renderobject.cuh"
#include "renderobjects/triangle.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/list.cuh"
#include "utils/settings.cuh"
#include "utils/utils.cuh"

class Octahedron : public RenderObject
{
  public:
    float size;

    Triangle *tris[8];

    Vector<float> x;
    Vector<float> y;
    Vector<float> z;

    __hd__ Octahedron();

    __hd__ Octahedron(const Vector<float> &origin, float size) : size(size)
    {
        this->origin = origin;
        gen_triangles();
    }

    __hd__ Octahedron(const Vector<float> &origin,
                      float size,
                      const Vector<int> &color,
                      float metallic,
                      float hardness,
                      float diffuse,
                      float specular,
                      float roughness)
        : size(size), RenderObject(color, metallic, hardness, diffuse, specular, roughness)
    {
        this->origin = origin;
        gen_triangles();
    }

    __hd__ void gen_triangles()
    {
        x = Vector<float>{size, 0, 0};
        y = Vector<float>{0, size, 0};
        z = Vector<float>{0, 0, size};

        // Bottom Half
        tris[0] = new Triangle(origin - y,
                               origin - x,
                               origin + z,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness);
        tris[1] = new Triangle{origin - y,
                               origin - z,
                               origin - x,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness};
        tris[2] = new Triangle{origin - y,
                               origin + x,
                               origin - z,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness};
        tris[3] = new Triangle{origin - y,
                               origin + z,
                               origin + x,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness};

        // Top Half
        tris[4] = new Triangle{origin + y,
                               origin + z,
                               origin - x,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness};
        tris[5] = new Triangle{origin + y,
                               origin + x,
                               origin + z,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness};
        tris[6] = new Triangle{origin + y,
                               origin - z,
                               origin + x,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness};
        tris[7] = new Triangle{origin + y,
                               origin - x,
                               origin - z,
                               color,
                               metallic,
                               hardness,
                               diffuse,
                               specular,
                               roughness};
    }

    __device__ void translate(Vector<float> translation)
    {
        for (int i = 0; i < 8; i++) {
            tris[i]->translate(translation);
        }
    }

    __device__ void rotate(RotationMatrix rotation)
    {
        for (int i = 0; i < 8; i++) {
            Triangle *curr = tris[i];
            curr->init(rotation * (curr->point0 - origin) + origin,
                       rotation * (curr->point1 - origin) + origin,
                       rotation * (curr->point2 - origin) + origin,
                       true);
        }
    }

    __hd__ bool is_visible(const Vector<float> &ray_origin,
                           const Vector<float> &ray,
                           Vector<float> &ray_collide_position,
                           Vector<float> &ray_reflect_direction,
                           Vector<float> &hit_normal,
                           float &hit_distance,

                           Vector<int> &object_color,
                           float &object_metallic,
                           float &object_hardness,
                           float &object_diffuse,
                           float &object_specular,
                           const Vector<float> &random_offsets) const
    {
        int closest_index = -1;
        float min_distance = C_INFINITY;
        for (int i = 0; i < 8; i++) {
            if (this->tris[i]->is_visible(ray_origin,
                                          ray,
                                          ray_collide_position,
                                          ray_reflect_direction,
                                          hit_normal,
                                          hit_distance,
                                          object_color,
                                          object_metallic,
                                          object_hardness,
                                          object_diffuse,
                                          object_specular,
                                          random_offsets)) {
                if (hit_distance < min_distance) {
                    min_distance = hit_distance;
                    closest_index = i;
                }
            }
        }
        return closest_index == -1 ? false
                                   : this->tris[closest_index]->is_visible(ray_origin,
                                                                           ray,
                                                                           ray_collide_position,
                                                                           ray_reflect_direction,
                                                                           hit_normal,
                                                                           hit_distance,
                                                                           object_color,
                                                                           object_metallic,
                                                                           object_hardness,
                                                                           object_diffuse,
                                                                           object_specular,
                                                                           random_offsets);
    }

    __hd__ void extend_list(List<RenderObject *> *list)
    {
        for (int i = 0; i < 8; i++) {
            list->add((RenderObject *)this->tris[i]);
        }
    }
};

#endif
