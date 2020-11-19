#pragma once

#include "shape.h"

class plane : public shape
{
public:
	plane(const float3 &pos, const float3 &orientation, const float3 &albedo, float metalness, float roughness, const float3 &emission)
	{
		this->type = PLANE;
		this->is_light = (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f);

		this->pos = pos;
		this->orientation = orientation;

		this->albedo = albedo;
		this->metalness = metalness;
		this->roughness = roughness;

		this->emission = emission;
	}


	void	d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) override
	{
		h_scene->obj[id].type = this->type;
		h_scene->obj[id].is_light = this->is_light;

		h_scene->obj[id].pos = this->pos;
		h_scene->obj[id].orientation = this->orientation;

		h_scene->obj[id].emission = this->emission;

		h_scene->obj[id].albedo_id = this->albedo_id;
		h_scene->obj[id].metalness_id = this->metalness_id;
		h_scene->obj[id].roughness_id = this->roughness_id;
		h_scene->obj[id].normal_id = this->normal_id;

		h_scene->obj[id].uv_scale = this->uv_scale;

		h_scene->obj[id].albedo = this->albedo;
		h_scene->obj[id].metalness = this->metalness;
		h_scene->obj[id].roughness = this->roughness;
	}
};
