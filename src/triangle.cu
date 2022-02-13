/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   11-Feb-2022
 * @brief Description
*/

#include "triangle.cuh"

void init_triangle(Triangle *tri,
                   const Vector<float> &point0,
                   const Vector<float> &point1,
                   const Vector<float> &point2)
{
    init_triangle(tri, point0, point1, point2, Vector<int>(255, 0, 0), 1.f);
}

void init_triangle(Triangle *tri,
                   const Vector<float> &point0,
                   const Vector<float> &point1,
                   const Vector<float> &point2,
                   const Vector<int> &color,
                   float metallic)
{
    tri->point0 = point0;
    tri->point1 = point1;
    tri->point2 = point2;

    tri->edge0 = point1 - point0;
    tri->edge1 = point2 - point0;

    tri->normal = (tri->edge1) ^ (tri->edge0);

    set_color(tri, color);
    set_metallic(tri, metallic);

    tri->base.type = TRIANGLE_ROT;
}

void set_color(Triangle *obj, const Vector<int> &new_color) { obj->base.color = new_color; }
void set_metallic(Triangle *obj, float &new_metallic) { obj->base.metallic = new_metallic; }

__hd__ bool is_visible(Triangle *tri,
                       const Vector<float> &ray_origin,
                       const Vector<float> &ray,
                       Vector<float> &ray_collide_position,
                       Vector<float> &ray_reflect_direction,
                       float &hit_distance,
                       Vector<int> &color,
                       float &object_reflectivity)
{

    if (tri->normal % ray >= 0) {
        return false;
    }

    // For brevity
    Vector<float> point0 = tri->point0;
    Vector<float> edge0  = tri->edge0;
    Vector<float> edge1  = tri->edge1;

    // Linear algebra for detecting collision
    float edge0_factor =
        (-(ray_origin.x - point0.x) * (ray.y * edge1.z - ray.z * edge1.y) +
         (ray_origin.y - point0.y) * (ray.x * edge1.z - ray.z * edge1.x) -
         (ray_origin.z - point0.z) * (ray.x * edge1.y - ray.y * edge1.x)) /
        (ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z +
         ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);
    float edge1_factor =
        ((ray_origin.x - point0.x) * (ray.y * edge0.z - ray.z * edge0.y) -
         (ray_origin.y - point0.y) * (ray.x * edge0.z - ray.z * edge0.x) +
         (ray_origin.z - point0.z) * (ray.x * edge0.y - ray.y * edge0.x)) /
        (ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z +
         ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);
    float ray_factor =
        (-(ray_origin.x - point0.x) * (edge0.y * edge1.z - edge0.z * edge1.y) +
         (ray_origin.y - point0.y) * (edge0.x * edge1.z - edge0.z * edge1.x) -
         (ray_origin.z - point0.z) * (edge0.x * edge1.y - edge0.y * edge1.x)) /
        (ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z +
         ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);

    if ((edge0_factor < 0 || edge0_factor > 1) || (edge1_factor < 0 || edge1_factor > 1) ||
        (edge0_factor + edge1_factor > 1) || ray_factor < 0) {
        return false;
    }

    hit_distance = (ray * ray_factor).length();

    if (hit_distance < HIT_PRECISION) {
        return false;
    }

    // Reflection
    ray_collide_position = point0 + (edge0 * edge0_factor) + (edge1 * edge1_factor);
	ray_reflect_direction = !(ray + !tri->normal * (!tri->normal % -ray) * 2);

    object_reflectivity = tri->base.metallic;

    color = tri->base.color;

    return true;
}
