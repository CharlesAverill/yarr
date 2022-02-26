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

    __hd__ Sphere(const Vector<float> center, float radius, Vector<int> color, float metallic)
        : center(center), radius(radius), RenderObject(color, metallic)
    {
    }

    Vector<float> center;
    float radius;

    __hd__ bool is_visible(const Vector<float> &ray_origin,
                           const Vector<float> &ray,
                           Vector<float> &ray_collide_position,
                           Vector<float> &ray_reflect_direction,
                           float &hit_distance,
                           Vector<int> &color,
                           float &object_metallic) const
    {
        Vector<float> p = this->center - ray_origin;
        float threshold = sqrt(p % p - this->radius * this->radius);
        float b = p % ray;

        if (b > threshold) {
            float s = (p % p) - (b * b);
            float t = (this->radius * this->radius) - (s * s);
            hit_distance = b - t;

            if (hit_distance < HIT_PRECISION) {
                return false;
            }

            ray_collide_position = ray_origin + ray * hit_distance;
            Vector<float> normal = !(-p + ray * hit_distance);
            ray_reflect_direction = !(ray + !normal * (!normal % -ray) * 2);

            color = this->color;
            object_metallic = this->metallic;

            return true;
        }

        return false;
    }
};

#endif
