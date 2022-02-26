/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   26-Feb-2022
 * @brief Description
*/

#include "renderobjects/sphere.cuh"

void init_sphere(Sphere *sph, const Vector<float> &center, const float radius)
{
    init_sphere(sph, center, radius, Vector<int>(255, 0, 0), 1.f);
}

void init_sphere(Sphere *sph,
                 const Vector<float> &center,
                 const float radius,
                 const Vector<int> &color,
                 float metallic)
{
    sph->center = center;
    sph->radius = radius;

    set_color(sph, color);
    set_metallic(sph, metallic);

    sph->base.type = SPHERE_ROT;
}

void set_color(Sphere *obj, const Vector<int> &new_color) { obj->base.color = new_color; }
void set_metallic(Sphere *obj, float &new_metallic) { obj->base.metallic = new_metallic; }

__hd__ bool is_visible(Sphere *sph,
                       const Vector<float> &ray_origin,
                       const Vector<float> &ray,
                       Vector<float> &ray_collide_position,
                       Vector<float> &ray_reflect_direction,
                       float &hit_distance,
                       Vector<int> &color,
                       float &object_reflectivity)
{
    Vector<float> p = sph->center - ray_origin;
    float threshold = sqrt(p % p - sph->radius * sph->radius);
    float b         = p % ray;

    if (b > threshold) {
        float s      = (p % p) - (b * b);
        float t      = (sph->radius * sph->radius) - (s * s);
        hit_distance = b - t;

        if (hit_distance < HIT_PRECISION) {
            return false;
        }

        ray_collide_position   = ray_origin + ray * hit_distance;
        Vector<float> normal   = !(-p + ray * hit_distance);
        ray_reflect_direction = !(ray + !normal * (!normal % -ray) * 2);

        color               = sph->base.color;
        object_reflectivity = sph->base.metallic;

        return true;
    }

    return false;
}
