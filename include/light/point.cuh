/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   26-Feb-2022
 * @brief Description
*/

#ifndef POINT_H
#define POINT_H

#include "linear_algebra/vector.cuh"
#include "utils/utils.cuh"

class Point
{
  public:
    Vector<float> position;
    Vector<int> color;

    __hd__ Point();

    __hd__ Point(Vector<float> position, Vector<int> color) : position(position), color(color)
    {
    }

    __hd__ float diffuse_at_point(Vector<float> normal, Vector<float> hit_position)
    {
        return max(0.f, normal % !(position - hit_position));
    }

    __hd__ float specular_at_point(Vector<float> hit_position, Vector<float> ray_direction)
    {
        return max(0.f, !(position - hit_position) % ray_direction);
    }
};

#endif
