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
    Vector<float> center;
    float size;

    Triangle *tris[8];

    __hd__ Octahedron();

    __hd__ Octahedron(const Vector<float> &center, float size) : center(center), size(size)
    {
        gen_triangles();
    }

    __hd__
    Octahedron(const Vector<float> &center, float size, const Vector<int> &color, float metallic)
        : center(center), size(size), RenderObject(color, metallic)
    {
        gen_triangles();
    }

    __hd__ void gen_triangles()
    {
        Vector<float> x = Vector<float>{size, 0, 0};
        Vector<float> y = Vector<float>{0, size, 0};
        Vector<float> z = Vector<float>{0, 0, size};

        // Bottom Half
        tris[0] = new Triangle{center - y, center - x, center + z, color, metallic};
        tris[1] = new Triangle{center - y, center - z, center - x, color, metallic};
        tris[2] = new Triangle{center - y, center + x, center - z, color, metallic};
        tris[3] = new Triangle{center - y, center + z, center + x, color, metallic};

        // Top Half
        tris[4] = new Triangle{center + y, center + z, center - x, color, metallic};
        tris[5] = new Triangle{center + y, center + z, center + x, color, metallic};
        tris[6] = new Triangle{center + y, center - z, center + x, color, metallic};
        tris[7] = new Triangle{center + y, center - x, center - z, color, metallic};
    }

    __hd__ bool is_visible(const Vector<float> &ray_origin,
                           const Vector<float> &ray,
                           Vector<float> &ray_collide_position,
                           Vector<float> &ray_reflect_direction,
                           float &hit_distance,
                           Vector<int> &color,
                           float &object_metallic) const
    {
        return false;
    }

    __hd__ void extend_list(List<RenderObject *> *list)
    {
        for (int i = 0; i < 8; i++) {
            list->add((RenderObject *)tris[i]);
        }
    }
};

#endif
