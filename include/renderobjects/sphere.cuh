/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   26-Feb-2022
 * @brief Description
*/

#ifndef SPHERE_H
#define SPHERE_H

#include "renderobjects/renderobject.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/settings.cuh"
#include "utils/utils.cuh"

class Sphere : public RenderObject
{
  public:
    __hd__ Sphere();

    __hd__ Sphere(const Vector<float> &center, float radius)
        : center(center), radius(radius), RenderObject()
    {
    }

    __hd__ Sphere(const Vector<float> center,
                  float radius,
                  Vector<int> color,
                  float metallic,
                  float hardness,
                  float diffuse,
                  float specular,
                  float roughness)
        : center(center), radius(radius),
          RenderObject(color, metallic, hardness, diffuse, specular, roughness)
    {
    }

    Vector<float> center;
    float radius;

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
        const Vector<float> p = center - ray_origin;
        const float threshold = std::sqrt(p % p - radius * radius);
        const float b = p % ray;

        if (b > threshold) {
            const float s = std::sqrt(p % p - b * b);
            const float t = std::sqrt(radius * radius - s * s);
            hit_distance = b - t;

            if (hit_distance < HIT_PRECISION)
                return false;

            ray_collide_position = ray_origin + ray * hit_distance;
            const Vector<float> normal = !(-p + ray * hit_distance);
            ray_reflect_direction = !(ray + !normal * (!normal % -ray) * 2);

            ray_reflect_direction = !(ray_reflect_direction + !random_offsets * roughness);

            if (normal % ray_reflect_direction <= 0) {
                ray_reflect_direction =
                    !(ray_reflect_direction + !normal * (!normal % -ray_reflect_direction) * 2);
            }

            object_color = color;
            object_metallic = metallic;
            object_hardness = hardness;
            object_diffuse = diffuse;
            object_specular = specular;

            //hit_normal = normal;

            return true;
        }

        return false;
    }
};

#endif
