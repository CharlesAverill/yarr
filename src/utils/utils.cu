/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "utils/utils.cuh"

template <typename T> T max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T> T min(T a, T b)
{
    return a < b ? a : b;
}
