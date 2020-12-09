#pragma once

#include <cuda_runtime.h>

#include <algorithm>


enum light_type
{
	SPHERICAL,
	DIRECTIONAL,
	LIGHT_TYPE_COUNT
};

class light
{
public:
	int			type;
	int			obj_id = -1;

	float3		emission;
	float		intensity;

	float3		dir;
	float		dir_factor;


	light(const float3 &emission, float intensity, int obj_id)
	{
		this->type = SPHERICAL;
		this->obj_id = obj_id;

		this->emission = emission;
		this->intensity = intensity;
	}

	light(const float3 &emission, float intensity, const float3 &dir, float dir_factor)
	{
		this->type = DIRECTIONAL;

		this->emission = emission;
		this->intensity = intensity;

		this->dir = h_normalize(dir);
		this->dir_factor = std::min(1.0f, std::max(0.0f, dir_factor));
	}


	void		d_malloc(gpu_scene *h_scene, const int light_id, cudaError_t &cuda_status)
	{
		h_scene->light[light_id].type = this->type;
		h_scene->light[light_id].obj_id = this->obj_id;

		h_scene->light[light_id].emission = this->emission;
		h_scene->light[light_id].intensity = this->intensity;

		h_scene->light[light_id].dir = this->dir;
		h_scene->light[light_id].dir_factor = this->dir_factor;
	}
};
