/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   25-Mar-2022
 * @brief Description
*/

#ifndef ROTATION_MATRIX_H
#define ROTATION_MATRIX_H

#include <math.h>

#include "linear_algebra/vector.cuh"
#include "utils/cuda_utils.cuh"

class RotationMatrix
{
  public:
    float data[3][3];

    __hd__ RotationMatrix(float x, float y, float z)
    {
        /*          A                               B                               C
           1   |    0    | 0             cos(y)|    0    |  sin(y)       cos(z)| -sin(z) |    0
        ---------------------------     ---------------------------     ---------------------------
           0   |  cos(x) | -sin(x)   x     0   |    1    |    0      x   sin(z)|  cos(x) |    0
        ---------------------------     ---------------------------     ---------------------------
           0   |  sin(x) |  cos(x)      -sin(y)|    0    |  cos(y)         0   |    0    |    1
        */
        *this = RotationMatrix(1, 0, 0, 0, cos(x), -sin(x), 0.0f, sin(x), cos(x)) *
                RotationMatrix(cos(y), 0.0f, sin(y), 0.0f, 1.0f, 0.0f, -sin(y), 0.0f, cos(y)) *
                RotationMatrix(cos(z), -sin(z), 0.0f, sin(z), cos(z), 0.0f, 0.0f, 0.0f, 1.0f);
    }

    __hd__
    RotationMatrix(float a, float b, float c, float d, float e, float f, float g, float h, float i)
    {
        /*
        a | b | c
        ---------
        d | e | f
        ---------
        g | h | i
        */
        data[0][0] = a;
        data[0][1] = b;
        data[0][2] = c;
        data[1][0] = d;
        data[1][1] = e;
        data[1][2] = f;
        data[2][0] = g;
        data[2][1] = h;
        data[2][2] = i;
    }

    __hd__ RotationMatrix operator*(const RotationMatrix &other) const
    {
        RotationMatrix out(0, 0, 0, 0, 0, 0, 0, 0, 0);

        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                out.data[y][x] = data[y][0] * other.data[0][x] + data[y][1] * other.data[1][x] +
                                 data[y][2] * other.data[2][x];
            }
        }

        return out;
    }

    __hd__ Vector<float> operator*(const Vector<float> &vec) const
    {
        return Vector<float>{data[0][0] * vec.x + data[0][1] * vec.y + data[0][2] * vec.z,
                             data[1][0] * vec.x + data[1][1] * vec.y + data[1][2] * vec.z,
                             data[2][0] * vec.x + data[2][1] * vec.y + data[2][2] * vec.z};
    }
};

#endif
