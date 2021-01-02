
#include "sphere.h"


sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, float emissive, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
	this->emissive = emissive;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, int roughness_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int normal_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int normal_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int normal_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, int roughness_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;
	this->emissive_id = emissive_id;
	this->emissive = emissive;

	this->uv_scale = uv_scale;
}
