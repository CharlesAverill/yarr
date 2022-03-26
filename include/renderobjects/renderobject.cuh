/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   11-Feb-2022
 * @brief Description
*/

#ifndef RENDEROBJECT_H
#define RENDEROBJECT_H

#include <nvfunctional>

#include "linear_algebra/rotation_matrix.cuh"
#include "linear_algebra/vector.cuh"

class RenderObject
{
  public:
    Vector<int> color;
    float metallic;
    float hardness;
    float diffuse;
    float specular;
    float roughness;

    Vector<float> origin;

    nvstd::function<void(int, RenderObject *)> update_lambda = nullptr;
    nvstd::function<Vector<float>(int, RenderObject *)> translation_lambda = nullptr;
    nvstd::function<RotationMatrix(int, RenderObject *)> rotation_lambda = nullptr;

    __hd__ RenderObject()
        : color(255, 255, 255), origin(0, 0, 0), metallic(1), hardness(1), diffuse(1), specular(1),
          roughness(0)
    {
    }

    __hd__ RenderObject(Vector<float> origin)
        : color(255, 255, 255), origin(origin), metallic(1), hardness(1), diffuse(1), specular(1),
          roughness(0)
    {
    }

    __hd__ RenderObject(Vector<int> color,
                        float metallic,
                        float hardness,
                        float diffuse,
                        float specular,
                        float roughness)
        : color(color), origin(0, 0, 0), metallic(metallic), hardness(hardness), diffuse(diffuse),
          specular(specular), roughness(roughness)
    {
    }

    __device__ void set_updates(const nvstd::function<Vector<float>(int, RenderObject *)> &t,
                                const nvstd::function<RotationMatrix(int, RenderObject *)> &r)
    {
        this->translation_lambda = t;
        this->rotation_lambda = r;
    }

    __device__ void set_update(const nvstd::function<void(int, RenderObject *)> &u)
    {
        this->update_lambda = u;
    }

    __device__ void
    set_translation_update(const nvstd::function<Vector<float>(int, RenderObject *)> &t)
    {
        this->translation_lambda = t;
    }

    __device__ void
    set_rotation_update(const nvstd::function<RotationMatrix(int, RenderObject *)> &r)
    {
        this->rotation_lambda = r;
    }

    __device__ virtual void translate(Vector<float> translation) = 0;
    __device__ virtual void rotate(RotationMatrix rotation) = 0;

    __device__ void update(int frame)
    {
        if (this->update_lambda != nullptr) {
            this->update_lambda(frame, this);
        } else {
            if (this->translation_lambda != nullptr) {
                this->translate(this->translation_lambda(frame, this));
            }
            if (this->rotation_lambda != nullptr) {
                this->rotate(this->rotation_lambda(frame, this));
            }
        }
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
                                   float &object_specular,
                                   const Vector<float> &random_offsets) const = 0;

    __hd__ void set_origin(const Vector<float> &v)
    {
        origin = v;
    }

    __hd__ void set_color(const Vector<int> &v)
    {
        color = v;
    }

    __hd__ void set_metallic(float v)
    {
        metallic = v;
    }

    __hd__ void set_hardness(float v)
    {
        hardness = v;
    }

    __hd__ void set_diffuse(float v)
    {
        diffuse = v;
    }

    __hd__ void set_specular(float v)
    {
        specular = v;
    }

    __hd__ void set_roughness(float v)
    {
        roughness = v;
    }
};

#endif
