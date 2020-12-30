#pragma once

#include "shape.h"


class triangle : public shape
{
private:
	void	set_triangle_props(const float3 vert[3], const float3 norm[3], const float2 uv[3])
	{
		this->type = TRIANGLE;
		this->is_light = 0;

		float3 a = vert[0];
		float3 b = vert[1];
		float3 c = vert[2];

		this->vert[0] = a;
		this->vert[1] = b;
		this->vert[2] = c;

		this->norm[0] = norm[0];
		this->norm[1] = norm[1];
		this->norm[2] = norm[2];

		this->uv[0] = uv[0];
		this->uv[1] = uv[1];
		this->uv[2] = uv[2];


		float3 da = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
		float3 db = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
		this->orientation = h_normalize(h_cross(da, db));
	}

public:
	float3	vert[3];
	float3	norm[3];
	float2	uv[3];


	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, float roughness);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, float roughness);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, float roughness);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, int roughness_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, int roughness_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, float roughness);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, int roughness_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, int roughness_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, float roughness, int normal_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, float roughness, int normal_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, float roughness, int normal_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, int roughness_id, int normal_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, int roughness_id, int normal_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, float roughness, int normal_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, int roughness_id, int normal_id);
	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, int roughness_id, int normal_id);

	triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], mesh_material_data mat_data)
	{
		this->set_triangle_props(vert, norm, uv);

		this->albedo_id = mat_data.albedo_id;
		this->metalness_id = mat_data.metalness_id;
		this->roughness_id = mat_data.roughness_id;
		this->normal_id = mat_data.normal_id;

		this->albedo = mat_data.albedo;
		this->metalness = mat_data.metalness;
		this->roughness = mat_data.roughness;
	}


	void	d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) override
	{
		h_scene->obj[id].type = this->type;
		h_scene->obj[id].is_light = this->is_light;

		h_scene->obj[id].vert[0] = this->vert[0];
		h_scene->obj[id].vert[1] = this->vert[1];
		h_scene->obj[id].vert[2] = this->vert[2];
		h_scene->obj[id].norm[0] = this->norm[0];
		h_scene->obj[id].norm[1] = this->norm[1];
		h_scene->obj[id].norm[2] = this->norm[2];
		h_scene->obj[id].uv[0] = this->uv[0];
		h_scene->obj[id].uv[1] = this->uv[1];
		h_scene->obj[id].uv[2] = this->uv[2];


		h_scene->obj[id].orientation = this->orientation;

		//h_scene->obj[id].emission = this->emission;

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
