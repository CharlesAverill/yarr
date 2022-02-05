/**
 * @file
 * @author Charles Averill
 * @date   05-Feb-2022
 * @brief Description
*/

#include "utils.cuh"

void hex_str_to_color_arr(int out[3], char in[6]) {
    long dec = strtol(in, NULL, 16);
    long mask1 = 255;
    long mask2 = 65280;
    long mask3 = 16711680;

    out[2] = dec & mask1;
    out[1] = (dec & mask2) >> 8;
    out[0] = (dec & mask3) >> 16;
}
