/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   11-Feb-2022
 * @brief Description
*/

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "renderobjects/renderobject.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/settings.cuh"
#include "utils/utils.cuh"

class Triangle : public RenderObject
{
  public:
    __hd__ Triangle();

    __hd__ Triangle(const Vector<float> &p0, const Vector<float> &p1, const Vector<float> &p2)
        : point0(p0), point1(p1), point2(p2), RenderObject()
    {
        edge0 = p1 - p0;
        edge1 = p2 - p0;
        normal = edge1 ^ edge0;
    }

    __hd__ Triangle(const Vector<float> &p0,
                    const Vector<float> &p1,
                    const Vector<float> &p2,
                    const Vector<int> &color,
                    float metallic,
                    float hardness,
                    float diffuse,
                    float specular,
                    float roughness)
        : point0(p0), point1(p1), point2(p2),
          RenderObject(color, metallic, hardness, diffuse, specular, roughness)
    {
        edge0 = p1 - p0;
        edge1 = p2 - p0;
        normal = edge1 ^ edge0;
    }

    Vector<float> point0;
    Vector<float> point1;
    Vector<float> point2;
    Vector<float> edge0;
    Vector<float> edge1;
    Vector<float> normal;

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
                           float &object_specular) const
    {
        if (this->normal % ray >= 0) {
            return false;
        }

        // For brevity
        Vector<float> point0 = this->point0;
        Vector<float> edge0 = this->edge0;
        Vector<float> edge1 = this->edge1;

        // Linear algebra for detecting collision
        float edge0_factor =
            (-(ray_origin.x - point0.x) * (ray.y * edge1.z - ray.z * edge1.y) +
             (ray_origin.y - point0.y) * (ray.x * edge1.z - ray.z * edge1.x) -
             (ray_origin.z - point0.z) * (ray.x * edge1.y - ray.y * edge1.x)) /
            (ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z +
             ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);
        float edge1_factor =
            ((ray_origin.x - point0.x) * (ray.y * edge0.z - ray.z * edge0.y) -
             (ray_origin.y - point0.y) * (ray.x * edge0.z - ray.z * edge0.x) +
             (ray_origin.z - point0.z) * (ray.x * edge0.y - ray.y * edge0.x)) /
            (ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z +
             ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);
        float ray_factor =
            (-(ray_origin.x - point0.x) * (edge0.y * edge1.z - edge0.z * edge1.y) +
             (ray_origin.y - point0.y) * (edge0.x * edge1.z - edge0.z * edge1.x) -
             (ray_origin.z - point0.z) * (edge0.x * edge1.y - edge0.y * edge1.x)) /
            (ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z +
             ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);

        if ((edge0_factor < 0 || edge0_factor > 1) || (edge1_factor < 0 || edge1_factor > 1) ||
            (edge0_factor + edge1_factor > 1) || ray_factor < 0) {
            return false;
        }

        hit_distance = (ray * ray_factor).length();

        if (hit_distance < HIT_PRECISION) {
            return false;
        }

        // Reflection
        ray_collide_position = point0 + (edge0 * edge0_factor) + (edge1 * edge1_factor);
        ray_reflect_direction = !(ray + !this->normal * (!this->normal % -ray) * 2);

        object_color = color;
        object_metallic = metallic;
        object_hardness = hardness;
        object_diffuse = diffuse;
        object_specular = specular;

        hit_normal = !normal;

        return true;
    }
};

#endif
