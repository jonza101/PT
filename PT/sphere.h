#pragma once

#include "shape.h"


class sphere : public shape
{
public:
	float		radius;


	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metallness, float roughness, const float3 &emission)
	{
		this->type = SPHERE;
		this->is_light = (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f);

		this->pos = pos;
		this->orientation = orientation;
		this->radius = radius;

		this->albedo = albedo;
		this->metallness = metallness;
		this->roughness = roughness;

		this->emission = emission;
	}

	void	d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) override
	{
		h_scene->obj[id].type = this->type;
		h_scene->obj[id].is_light = this->is_light;

		h_scene->obj[id].pos = this->pos;
		h_scene->obj[id].orientation = this->orientation;
		h_scene->obj[id].radius = this->radius;

		h_scene->obj[id].emission = this->emission;

		h_scene->obj[id].albedo = this->albedo;
		h_scene->obj[id].metallness = this->metallness;
		h_scene->obj[id].roughness = this->roughness;
	}
};
