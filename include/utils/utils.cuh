/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#ifndef UTILS_H
#define UTILS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define C_INFINITY 0x7FF0000000000000  // Infinity
#define C_NINFINITY 0xFFF0000000000000 // Negative Infinity
#define C_BILLION 1000000000.f

template <typename T> T max(T a, T b);
template <typename T> T min(T a, T b);

#endif
