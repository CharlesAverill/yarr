/**
 * @file
 * @author Charles Averill <charlesaverill>
 * @date   12-Feb-2022
 * @brief Description
*/

// Framerate
#define TARGET_FPS 60

// Distance determining whether a ray hits a triangle or not
#define HIT_PRECISION 0.001f

// Max number of reflections per ray
#define MAX_REFLECTIONS 100

// Ambient lighting in scene
#define AMBIENT_LIGHT 0.3f

// Ground settings
#define GROUND_METALLIC 0.f
#define GROUND_HARDNESS 0.f
#define GROUND_DIFFUSE 0.8f
#define GROUND_SPECULAR 0.f

// Anti-aliasing
#define DO_ANTIALIASING true
#define ANTIALIASING_SAMPLES 16

// Soft Shading
#define SOFT_SHADOW_FACTOR 15

// Depth of field
#define DOF 0.01f

// RNG
#define SEED 1234
