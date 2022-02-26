/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   11-Feb-2022
 * @brief Description
*/

#ifndef RENDEROBJECT_H
#define RENDEROBJECT_H

#include "utils/vector.cuh"

enum RenderObjectType {
    RENDEROBJECT_ROT,
    TRIANGLE_ROT,
    SPHERE_ROT,
};

typedef struct {
    Vector<int> color;
    float metallic;

    RenderObjectType type;
} RenderObject;

void set_color(RenderObject *obj, const Vector<int> &new_color);
void set_metallic(RenderObject *obj, float metallic);

#endif
