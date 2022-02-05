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

template <typename T>
class Vector {
public:
    // Values
    T x;
    T y;
    T z;

    // Constructors
    __hd__ Vector(T a=0.0f, T b=0.0f, T c=0.0f) {
        init(a, b, c);
    }

    __hd__ void init(T a=0.0f, T b=0.0f, T c=0.0f) {
        x = a;
        y = b;
        z = c;
    }

    // Copy constructor
    Vector(const Vector &other) {
        x = other.x;
        y = other.y;
        z = other.z;
    }

    // Getters
    __hd__ float length() const;

    // Operator overloading
    // Addition
    __hd__ Vector operator+(const Vector& other) const;
    // Negation
    __hd__ Vector operator-() const;
    // Scalar multiplication
    __hd__ Vector operator*(T r) const;
    // Dot product
    __hd__ T operator%(const Vector& other) const;
    // Cross product
    __hd__ Vector operator^(const Vector& other) const;
    // Unit vector
    __hd__ Vector operator!() const;
};

#endif
