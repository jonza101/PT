#pragma once

#include "shape.h"


class sphere : public shape
{
private:
	void		set_sphere_props(const float3 &pos, const float3 &orientation, float radius, bool is_light, float intensity)
	{
		this->type = SPHERE;
		this->is_light = is_light;

		this->pos = pos;
		this->orientation = orientation;
		this->radius = radius;

		this->intensity = intensity;
	}

public:
	float		radius;


	sphere(const float3 &pos, const float3 &orientation, float radius, material_data mat_data, bool is_light = false, float intensity = 0.0f)
	{
		this->type = SPHERE;
		this->is_light = is_light;

		this->pos = pos;
		this->orientation = orientation;
		this->radius = radius;

		this->intensity = intensity;

		this->albedo_id = mat_data.albedo_id;
		this->metalness_id = mat_data.metalness_id;
		this->roughness_id = mat_data.roughness_id;
		this->reflectance_id = mat_data.reflectance_id;
		this->ior_id = mat_data.ior_id;
		this->normal_id = mat_data.normal_id;
		this->emissive_id = mat_data.emissive_id;

		this->albedo = mat_data.albedo;
		this->metalness = mat_data.metalness;
		this->roughness = mat_data.roughness;
		this->reflectance = mat_data.reflectance;
		this->ior = mat_data.ior;
		this->emissive = mat_data.emissive;

		this->uv_scale = mat_data.uv_scale;
	}


	void	d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) override
	{
		h_scene->obj[id].type = this->type;
		h_scene->obj[id].is_light = this->is_light;

		h_scene->obj[id].pos = this->pos;
		h_scene->obj[id].orientation = this->orientation;
		h_scene->obj[id].radius = this->radius;

		h_scene->obj[id].albedo_id = this->albedo_id;
		h_scene->obj[id].metalness_id = this->metalness_id;
		h_scene->obj[id].roughness_id = this->roughness_id;
		h_scene->obj[id].reflectance_id = this->reflectance_id;
		h_scene->obj[id].ior_id = this->ior_id;
		h_scene->obj[id].normal_id = this->normal_id;
		h_scene->obj[id].emissive_id = this->emissive_id;

		h_scene->obj[id].albedo = this->albedo;
		h_scene->obj[id].metalness = this->metalness;
		h_scene->obj[id].roughness = this->roughness;
		h_scene->obj[id].reflectance = this->reflectance;
		h_scene->obj[id].ior = this->ior;
		h_scene->obj[id].emissive = this->emissive;

		h_scene->obj[id].uv_scale = this->uv_scale;
	}
};
