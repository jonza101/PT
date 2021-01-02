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
	//p_albedo, 0.01f, 0.5f, p_normal, mt_roughness, 0.25f,

	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, float emissive, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, int roughness_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int normal_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int normal_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);

	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);
	sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light = false, float intensity = 0.0f);


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
		h_scene->obj[id].normal_id = this->normal_id;
		h_scene->obj[id].emissive_id = this->emissive_id;

		h_scene->obj[id].uv_scale = this->uv_scale;

		h_scene->obj[id].albedo = this->albedo;
		h_scene->obj[id].metalness = this->metalness;
		h_scene->obj[id].roughness = this->roughness;
		h_scene->obj[id].emissive = this->emissive;
	}
};
