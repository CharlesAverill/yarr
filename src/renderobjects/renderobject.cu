/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   12-Feb-2022
 * @brief Description
*/

#include "renderobjects/renderobject.cuh"

void set_color(RenderObject *obj, const Vector<int> &new_color) { obj->color = new_color; }
void set_metallic(RenderObject *obj, float metallic) { obj->metallic = metallic; }
