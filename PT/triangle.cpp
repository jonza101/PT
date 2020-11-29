
#include "triangle.h"


triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, float roughness)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, float roughness)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, float roughness)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, int roughness_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, int roughness_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, float roughness)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, int roughness_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, int roughness_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, float roughness, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, float roughness, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, float roughness, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, int metalness_id, int roughness_id, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], int albedo_id, float metalness, int roughness_id, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, float roughness, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, int metalness_id, int roughness_id, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
}

triangle::triangle(const float3 vert[3], const float3 norm[3], const float2 uv[3], const float3 &albedo, float metalness, int roughness_id, int normal_id)
{
	this->set_triangle_props(vert, norm, uv);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
}
