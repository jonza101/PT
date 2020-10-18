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

	//	Rz Ry Rx
	float3		right;				//	PITCH
	float3		up;					//	YAW
	float3		forward;			//	ROLL

	float		pitch;
	float		yaw;
	float		roll;


	camera()
	{
		this->pos = make_float3(0.0f, 0.0f, 0.0f);
		this->fov_deg = 90.0f;
		this->z_near = 0.0f;
		this->z_far = 1000.0f;

		this->up = make_float3(0.0f, 1.0f, 0.0f);
		this->forward = make_float3(0.0f, 0.0f, 1.0f);
		this->right = h_normalize(h_cross(this->forward, this->up));
		this->right.x *= -1.0f;
		this->right.y *= -1.0f;
		this->right.z *= -1.0f;

		this->calc_rotation_matrix();
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
		this->up = h_normalize(h_cross(this->forward, this->right));

		this->calc_rotation_matrix();

		/*std::cout << "f " << this->forward.x << ' ' << this->forward.y << ' ' << this->forward.z << '\n';
		std::cout << "r " << this->right.x << ' ' << this->right.y << ' ' << this->right.z << '\n';
		std::cout << "u " << this->up.x << ' ' << this->up.y << ' ' << this->up.z << '\n';*/
	}

	void	calc_rotation_matrix()
	{
		this->pitch = atan((float)this->up.z / ((float)this->forward.z + 0.01f));
		this->yaw = asinf(-this->right.z);
		this->roll = atan((float)this->right.y / ((float)this->right.x + 0.01f));
	}

	void	transform_ray_dir(float3 &dir)
	{
		//	Z
		dir.x = dir.x * cosf(this->roll) - dir.x * sinf(this->roll);
		dir.y = dir.y * sinf(this->roll) + dir.y * cosf(this->roll);
		dir.z = dir.z;

		//	Y
		dir.x = dir.x * cosf(this->yaw) + dir.z * sinf(this->yaw);
		dir.y = dir.y;
		dir.z = dir.x * sinf(this->yaw) + dir.z * cosf(this->yaw);

		//	X
		dir.x = dir.x;
		dir.y = dir.y * cosf(this->pitch) - dir.z * sinf(this->pitch);
		dir.z = dir.y * sinf(this->pitch) + dir.z * cosf(this->pitch);

		dir = h_normalize(dir);

		//std::cout << dir.x << ' ' << dir.y << ' ' << dir.z << '\n';
	}
};
