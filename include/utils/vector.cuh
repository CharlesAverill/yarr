/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef VECTOR_H
#define VECTOR_H

#include <stdexcept>

#include "cuda_utils.cuh"

template <typename T> class Vector
{
  public:
    // Values
    T x;
    T y;
    T z;

    // Constructors
    __hd__ Vector(T a = 0, T b = 0, T c = 0)
    {
        init(a, b, c);
    }

    __hd__ void init(T a = 0, T b = 0, T c = 0)
    {
        x = a;
        y = b;
        z = c;
    }

    // Copy constructor
    __hd__ Vector(const Vector &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }

    // Getters
    __hd__ float length() const
    {
        return sqrt((x * x) + (y * y) + (z * z));
    }

    // Operator overloading
    // Addition
    __hd__ Vector<T> operator+(const Vector &other) const
    {
        return Vector<T>(x + other.x, y + other.y, z + other.z);
    }
    // Subtraction
    __hd__ Vector<T> operator-(const Vector &other) const
    {
        return Vector<T>(x - other.x, y - other.y, z - other.z);
    }
    // Negation
    __hd__ Vector<T> operator-() const
    {
        return (*this) * -1;
    }
    // Scalar multiplication
    __hd__ Vector<T> operator*(float r) const
    {
        return Vector<T>(float(x) * r, float(y) * r, float(z) * r);
    }
    // Dot product
    __hd__ T operator%(const Vector &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }
    // Cross product
    __hd__ Vector<T> operator^(const Vector &other) const
    {
        return Vector<T>((y * other.z) - (z * other.y),
                         (z * other.x) - (x * other.z),
                         (x * other.y) - (y * other.x));
    }
    // Unit vector
    __hd__ Vector<T> operator!() const
    {
        return (*this) * (1.0f / this->length());
    }
};

#endif
