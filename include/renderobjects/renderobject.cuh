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
    Vector<int> color;
    float metallic;
    float hardness;
    float diffuse;
    float specular;
    float roughness;

    __hd__ RenderObject()
        : color(255, 255, 255), metallic(1), hardness(1), diffuse(1), specular(1), roughness(0)
    {
    }

    __hd__ RenderObject(Vector<int> color,
                        float metallic,
                        float hardness,
                        float diffuse,
                        float specular,
                        float roughness)
        : color(color), metallic(metallic), hardness(hardness), diffuse(diffuse),
          specular(specular), roughness(roughness)
    {
    }

    __hd__ virtual bool is_visible(const Vector<float> &ray_origin,
                                   const Vector<float> &ray,
                                   Vector<float> &ray_collide_position,
                                   Vector<float> &ray_reflect_direction,
                                   Vector<float> &hit_normal,
                                   float &hit_distance,

                                   Vector<int> &object_color,
                                   float &object_metallic,
                                   float &object_hardness,
                                   float &object_diffuse,
                                   float &object_specular) const = 0;

    void set_color(const Vector<int> &v)
    {
        color = v;
    }

    void set_metallic(float v)
    {
        metallic = v;
    }

    void set_hardness(float v)
    {
        hardness = v;
    }

    void set_diffuse(float v)
    {
        diffuse = v;
    }

    void set_specular(float v)
    {
        specular = v;
    }

    void set_roughness(float v)
    {
        roughness = v;
    }
};

#endif
