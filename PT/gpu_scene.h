#pragma once

#include <cuda_runtime.h>


struct				gpu_cam
{
	float3			pos;

	float			pitch;
	float			yaw;
	float			roll;

	float			aspect_ratio;
	float			scale;

	float			z_near;
	float			z_far;
};


struct				d_light_data
{
	int				obj_id;

	float3			pos;

	float3			emission;
	float			constant;
	float			linear;
	float			quadratic;
};

struct				d_obj_data
{
	int				type;
	int				is_light;

	float3			pos;
	float3			orientation;

	float3			emission;

	float3			albedo;
	float			metallness;
	float			roughness;


	float			radius;
};

struct				gpu_scene
{
	int				obj_count;
	d_obj_data		*obj;

	int				light_count;
	d_light_data	*light;
};
