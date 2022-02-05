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

#include "canvas.cuh"
#include "vector.cuh"

void hex_str_to_color_vec(Vector<int> *out, char in[6]);

#endif
