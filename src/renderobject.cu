/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   12-Feb-2022
 * @brief Description
*/

#include "renderobject.cuh"

void set_color(RenderObject *obj, const Vector<int> &new_color) {
    obj->color = new_color;
}
