/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   11-Feb-2022
 * @brief Description
*/

#ifndef RENDEROBJECT_H
#define RENDEROBJECT_H

#include "vector.cuh"

enum RenderObjectType {
    RENDEROBJECT_ROT,
    TRIANGLE_ROT,
};

typedef struct {
    Vector<int> color;
    RenderObjectType type;
} RenderObject;

void set_color(RenderObject *obj, const Vector<int> &new_color);

#endif
