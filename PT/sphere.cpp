
#include "sphere.h"


//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness = metalness;
//	this->roughness = roughness;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness = metalness;
//	this->roughness = roughness;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness_id = metalness_id;
//	this->roughness = roughness;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness_id = metalness_id;
//	this->roughness_id = roughness_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness = metalness;
//	this->roughness_id = roughness_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness_id = metalness_id;
//	this->roughness = roughness;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness_id = metalness_id;
//	this->roughness_id = roughness_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, int roughness_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness = metalness;
//	this->roughness_id = roughness_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness = metalness;
//	this->roughness = roughness;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness = metalness;
//	this->roughness = roughness;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness_id = metalness_id;
//	this->roughness = roughness;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness_id = metalness_id;
//	this->roughness_id = roughness_id;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo_id = albedo_id;
//	this->metalness = metalness;
//	this->roughness_id = roughness_id;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness_id = metalness_id;
//	this->roughness = roughness;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness_id = metalness_id;
//	this->roughness_id = roughness_id;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}
//
//sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, int roughness_id, int normal_id, const float2 &uv_scale, const float3 &emission, float intensity)
//{
//	this->set_sphere_props(pos, orientation, radius, emission, intensity);
//
//	this->albedo = albedo;
//	this->metalness = metalness;
//	this->roughness_id = roughness_id;
//	this->normal_id = normal_id;
//
//	this->uv_scale = uv_scale;
//}





sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, int roughness_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness_id = roughness_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, float metalness, float roughness, int normal_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, float roughness, int normal_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness = roughness;
	this->normal_id = normal_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, float roughness, int normal_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, int metalness_id, int roughness_id, int normal_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, int albedo_id, float metalness, int roughness_id, int normal_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo_id = albedo_id;
	this->metalness = metalness;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, float roughness, int normal_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness = roughness;
	this->normal_id = normal_id;

	this->uv_scale = uv_scale;
}

sphere::sphere(const float3 &pos, const float3 &orientation, float radius, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, const float2 &uv_scale, bool is_light, float intensity)
{
	this->set_sphere_props(pos, orientation, radius, is_light, intensity);

	this->albedo = albedo;
	this->metalness_id = metalness_id;
	this->roughness_id = roughness_id;
	this->normal_id = normal_id;

	this->uv_scale = uv_scale;
}
