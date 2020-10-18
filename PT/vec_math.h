#pragma once

#include <cuda_runtime.h>
#include <math.h>


float		h_length(const float3 &a);
float		h_dot(const float3 &a, const float3 &b);
float3		h_cross(const float3 &a, const float3 &b);
float3		h_normalize(const float3 &a);
