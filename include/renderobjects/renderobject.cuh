/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   11-Feb-2022
 * @brief Description
*/

#ifndef RENDEROBJECT_H
#define RENDEROBJECT_H

#include "utils/vector.cuh"

class RenderObject
{
  public:
    __hd__ RenderObject() : color(255, 255, 255), metallic(1)
    {
    }

    __hd__ RenderObject(Vector<int> color, float metallic) : color(color), metallic(metallic)
    {
    }

    __hd__ virtual bool is_visible(const Vector<float> &ray_origin,
                                   const Vector<float> &ray,
                                   Vector<float> &ray_collide_position,
                                   Vector<float> &ray_reflect_direction,
                                   float &hit_distance,
                                   Vector<int> &color,
                                   float &object_metallic) const = 0;

    void set_color(const Vector<int> &v)
    {
        color = v;
    }

    void set_metallic(float v)
    {
        metallic = v;
    }

    Vector<int> color;
    float metallic;
};

#endif
