/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   26-Feb-2022
 * @brief Description
*/

#ifndef SPHERE_H
#define SPHERE_H

#include "renderobjects/renderobject.cuh"
#include "settings.cuh"
#include "utils.cuh"

typedef struct {
    RenderObject base;

    Vector<float> center;
    float radius;
} Sphere;

void init_sphere(Sphere *sph, const Vector<float> &center, const float radius);

void init_sphere(Sphere *sph,
                 const Vector<float> &center,
                 const float radius,
                 const Vector<int> &color,
                 float metallic);

void set_color(Sphere *obj, const Vector<int> &new_color);
void set_metallic(Sphere *obj, float &new_metallic);

__hd__ bool is_visible(Sphere *sph,
                       const Vector<float> &ray_origin,
                       const Vector<float> &ray,
                       Vector<float> &ray_collide_position,
                       Vector<float> &ray_reflect_direction,
                       float &hit_distance,
                       Vector<int> &color,
                       float &object_reflectivity);

#endif
