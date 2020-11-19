
#ifndef VEC_MATH_CUH
#define VEC_MATH_CUH

#include "pt.h"


__device__ float	length(const float3 &a);
__device__ float	dot(const float3 &a, const float3 &b);
__device__ float3	cross(const float3 &a, const float3 &b);
__device__ float3	normalize(const float3 &a);

__device__ float3	lerp(const float3 &a, const float3 &b, float factor);
__device__ float3	reflect(const float3 &i, const float3 &n);
__device__ float3	refract(const float3 &i, const float3 &n, float etai, float etat);

#endif
