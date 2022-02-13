/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   11-Feb-2022
 * @brief Description
*/

#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "renderobject.cuh"
#include "utils.cuh"
#include "settings.cuh"

typedef struct {
    RenderObject base;

    Vector<float> point0;
    Vector<float> point1;
    Vector<float> point2;

    Vector<float> edge0;
    Vector<float> edge1;

    Vector<float> normal;
} Triangle;

void init_triangle(Triangle *tri,
                   const Vector<float> &point0,
                   const Vector<float> &point1,
                   const Vector<float> &point2);

void init_triangle(Triangle *tri,
                   const Vector<float> &point0,
                   const Vector<float> &point1,
                   const Vector<float> &point2,
                   const Vector<int> &color,
                   float metallic);

void set_color(Triangle *obj, const Vector<int> &new_color);
void set_metallic(Triangle *obj, float &new_metallic);

__hd__ bool is_visible(Triangle *tri,
                       const Vector<float> &ray_origin,
                       const Vector<float> &ray,
                       Vector<float> &ray_collide_position,
                       Vector<float> &ray_reflect_direction,
                       float &hit_distance,
                       Vector<int> &color,
                       float &object_reflectivity);

#endif
