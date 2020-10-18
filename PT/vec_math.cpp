
#include "vec_math.h"


float		h_length(const float3 &a)
{
	return (sqrtf(a.x * a.x + a.y * a.y + a.z * a.z));
}

float		h_dot(const float3 &a, const float3 &b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

float3		h_cross(const float3 &a, const float3 &b)
{
	return (make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x));
}

float3		h_normalize(const float3 &a)
{
	float3 v;

	float len = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	v.x = (float)a.x / (float)len;
	v.y = (float)a.y / (float)len;
	v.z = (float)a.z / (float)len;

	return (v);
}
