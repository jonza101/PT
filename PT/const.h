#pragma once

#define EPSILON 0.000001f
#define RAY_OFFSET 0.0001f

#define D_PDF_CONST ((float)1.0 / (float)(2.0f * M_PI))
#define HDR_CONST (1.0f / 2.2f)

#define MAX_BOUNCES 10
#define CUTOFF_THRESHOLD 5


#define BOUNDING_PLANES 7

#define BNC ((float)sqrtf(3.0f) / 3.0f)
const float3 BVH_NORMALS[BOUNDING_PLANES] = { make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(BNC, BNC, BNC), make_float3(-BNC, BNC, BNC), make_float3(-BNC, -BNC, BNC), make_float3(BNC, -BNC, BNC) };
