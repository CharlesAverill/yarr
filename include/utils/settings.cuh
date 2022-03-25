/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   12-Feb-2022
 * @brief Description
*/

#ifndef SETTINGS_CUH
#define SETTINGS_CUH

// clang-format off

// Framerate
#define TARGET_FPS 60

// Distance determining whether a ray hits a triangle or not
#define HIT_PRECISION 0.001f

// Max number of reflections per ray
#define MAX_REFLECTIONS 100

// Ambient lighting in scene
#define AMBIENT_LIGHT 0.3f

// Ground settings
#define GROUND_METALLIC 0.1f
#define GROUND_HARDNESS 0.f
#define GROUND_DIFFUSE 0.8f
#define GROUND_SPECULAR 0.1f

// Defines the style of the ground plane
enum GROUND_TYPES {
    GT_PLAIN,   // Solid color using GROUND_PRIMARY_COLOR
    GT_CHECKER, // A checkerboard of GROUND_PRIMARY_COLOR and GROUND_SECONDARY_COLOR
    GT_HSTRIPE, // Stripes of GROUND_PRIMARY_COLOR and GROUND_SECONDARY_COLOR running along the Z axis
    GT_VSTRIPE, // Stripes of GROUND_PRIMARY_COLOR and GROUND_SECONDARY_COLOR running along the X axis
    GT_RINGS,   // Concentric rings of GROUND_PRIMARY_COLOR and GROUND_SECONDARY_COLOR surrounding the origin
    GT_TEXTURE, // Uses GROUND_TEXTURE to render a repeated image onto the plane
};

#define GROUND_TYPE GT_TEXTURE

#define GROUND_PRIMARY_COLOR 0xFF0000
#define GROUND_SECONDARY_COLOR 0xFFFFFF

#define GROUND_TEXTURE_PATH ASSETS_ROOT "textures/cobble_stone_1.png"

// Anti-aliasing
#define DO_ANTIALIASING true
#define ANTIALIASING_SAMPLES 16

// Soft Shading
#define SOFT_SHADOW_FACTOR 15

// Depth of field
#define DOF 0.01f

// RNG
#define SEED 1234

// clang-format off

#endif
