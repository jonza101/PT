#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>

#include "gpu_scene.h"


enum obj_type
{
	SPHERE,
	PLANE,
	SHAPE_TYPE_COUNT
};

class shape
{
public:
	int				type;
	int				is_light;

	float3			pos;
	float3			orientation;

	int				albedo_id = -1;
	int				metalness_id = -1;
	int				roughness_id = -1;
	int				normal_id = -1;

	float3			albedo;
	float			metalness;
	float			roughness;

	float2			uv_scale;

	float3			emission;
	float			intensity;


	virtual void	d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) = 0;


	void			d_malloc_light(gpu_scene *h_scene, const int obj_id, const int light_id, cudaError_t &cuda_status)
	{
		h_scene->light[light_id].obj_id = obj_id;

		h_scene->light[light_id].pos = this->pos;
		h_scene->light[light_id].emission = this->emission;
		h_scene->light[light_id].intensity = this->intensity;
	}
};
