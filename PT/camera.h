#pragma once

//#include <cuda_runtime.h>

#include "vec_math.h"


class camera
{
public:
	float3		pos;
	
	float		fov_deg;

	float		z_near;
	float		z_far;

	float3		right;
	float3		up;
	float3		forward;


	camera()
	{
		this->pos = make_float3(0.0f, 0.0f, 0.0f);
		this->fov_deg = 70.0f;
		this->z_near = 0.0f;
		this->z_far = 1000.0f;

		this->up = make_float3(0.0f, 1.0f, 0.0f);
		this->forward = make_float3(0.0f, 0.0f, 1.0f);
		this->right = h_normalize(h_cross(this->forward, this->up));
		this->right.x *= -1.0f;
		this->right.y *= -1.0f;
		this->right.z *= -1.0f;
	}

	camera(const float3 &pos, float fov_deg, const float3 &forward, float z_near, float z_far)
	{
		this->pos = pos;
		this->fov_deg = fov_deg;
		this->z_near = z_near;
		this->z_far = z_far;

		this->up = make_float3(0.0f, 1.0f, 0.0f);
		this->forward = h_normalize(forward);
		this->right = h_normalize(h_cross(this->forward, this->up));
		this->right.x *= -1.0f;
		this->right.y *= -1.0f;
		this->right.z *= -1.0f;
		this->up = h_normalize(h_cross(this->right, this->forward));
	}
};
