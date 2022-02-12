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
                   const Vector<float> &point2) {
    init_triangle(tri, point0, point1, point2, Vector<int>(255, 0, 0));
}

void init_triangle(Triangle *tri,
                   const Vector<float> &point0,
                   const Vector<float> &point1,
                   const Vector<float> &point2,
                   const Vector<int> &color) {
    tri->point0 = point0;
    tri->point1 = point1;
    tri->point2 = point2;

    tri->edge0 = point1 - point0;
    tri->edge1 = point2 - point0;
    printf("%f %f %f\n", tri->edge0.x, tri->edge0.y, tri->edge0.z);
    printf("%f %f %f\n", tri->edge1.x, tri->edge1.y, tri->edge1.z);

    tri->normal = (tri->edge1) ^ (tri->edge0);
    printf("%f %f %f\n", tri->normal.x, tri->normal.y, tri->normal.z);

    set_color(tri, color);

    tri->base.type = TRIANGLE_ROT;
}

void set_color(Triangle *obj, const Vector<int> &new_color) {
    obj->base.color = new_color;
}

__hd__ bool is_visible(Triangle *tri,
                       const Vector<float> &ray_origin,
                       const Vector<float> &ray,
                       float &hit_distance,
                       Vector<int> &color) {

    if(tri->normal % ray >= 0) {
        return false;
    }

    Vector<float> point0 = tri->point0;
    Vector<float> edge0 = tri->edge0;
    Vector<float> edge1 = tri->edge1;

    float edge0_term = (-(ray_origin.x - point0.x) * (ray.y * edge1.z - ray.z * edge1.y) + (ray_origin.y - point0.y) * (ray.x * edge1.z - ray.z * edge1.x) - (ray_origin.z - point0.z) * (ray.x * edge1.y - ray.y * edge1.x))/(ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z + ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);
	float edge1_term = ((ray_origin.x - point0.x) * (ray.y * edge0.z - ray.z * edge0.y) - (ray_origin.y - point0.y) * (ray.x * edge0.z - ray.z * edge0.x) + (ray_origin.z - point0.z) * (ray.x * edge0.y - ray.y * edge0.x))/(ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z + ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);
	float ray_term = (-(ray_origin.x - point0.x) * (edge0.y * edge1.z - edge0.z * edge1.y) + (ray_origin.y - point0.y) * (edge0.x * edge1.z - edge0.z * edge1.x) - (ray_origin.z - point0.z) * (edge0.x * edge1.y - edge0.y * edge1.x))/(ray.x * edge0.y * edge1.z - ray.x * edge0.z * edge1.y - ray.y * edge0.x * edge1.z + ray.y * edge0.z * edge1.x + ray.z * edge0.x * edge1.y - ray.z * edge0.y * edge1.x);

    if ((edge0_term < 0 || edge0_term > 1) ||
		(edge1_term < 0 || edge1_term > 1) ||
		(edge1_term + edge1_term > 1) ||
		ray_term < 0) {
        return false;
    }

    hit_distance = (ray * ray_term).length();

    if (hit_distance < 0.001) {
        return false;
    }

    color = tri->base.color;

    return true;
}
