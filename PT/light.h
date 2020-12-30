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

	//float3		emission;
	float		intensity;

	int			albedo_id = -1;
	int			metalness_id = -1;
	int			roughness_id = -1;
	int			normal_id = -1;

	float3		albedo;
	float		metalness;
	float		roughness;

	float2		uv_scale;

	//	DIR LIGHT
	float3		dir;
	float		dir_factor;

public:
	light(const float3 &albedo, int albedo_id, float metalness, int metalness_id, float roughness, int roughness_id, int normal_id, const float2 &uv_scale, float intensity, int obj_id)
	{
		this->type = SPHERICAL;
		this->obj_id = obj_id;

		this->intensity = intensity;

		this->albedo_id = albedo_id;
		this->metalness_id = metalness_id;
		this->roughness_id = roughness_id;
		this->normal_id = normal_id;

		this->albedo = albedo;
		this->metalness = metalness;
		this->roughness = roughness;
	}


	//	DIR LIGHT
	light(const float3 &albedo, float intensity, const float3 &dir, float dir_factor)
	{
		this->type = DIRECTIONAL;

		this->albedo = albedo;
		this->intensity = intensity;

		this->dir = h_normalize(dir);
		this->dir_factor = std::min(1.0f, std::max(0.0f, dir_factor));
	}


	void		d_malloc(gpu_scene *h_scene, const int light_id, cudaError_t &cuda_status)
	{
		h_scene->light[light_id].type = this->type;
		h_scene->light[light_id].obj_id = this->obj_id;

		h_scene->light[light_id].albedo = this->albedo;
		h_scene->light[light_id].intensity = this->intensity;

		h_scene->light[light_id].dir = this->dir;
		h_scene->light[light_id].dir_factor = this->dir_factor;
	}
};
