/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "vector.cuh"

template <typename T>
__hd__ float Vector<T>::length() const {
    return sqrt((x * x) + (y * y) + (z * z));
}

template <typename T>
__hd__ Vector<T> Vector<T>::operator+(const Vector& other) const {
    return Vector(x + other.x, y + other.y, z + other.z);
}

template <typename T>
__hd__ Vector<T> Vector<T>::operator-() const {
    return (*this) * -1;
}

template <typename T>
__hd__ Vector<T> Vector<T>::operator*(T r) const {
    return Vector(x * r, y * r, z * r);
}

template <typename T>
__hd__ T Vector<T>::operator%(const Vector& other) const {
    return x * other.x + y * other.y + z * other.z;
}

template <typename T>
__hd__ Vector<T> Vector<T>::operator^(const Vector& other) const {
    return Vector((y * other.z) - (z * other.y),
                  (z * other.x) - (x * other.z),
                  (x * other.y) - (y * other.x));
}

template <typename T>
__hd__ Vector<T> Vector<T>::operator!() const {
    return (*this) * (1.0f / length());
}
