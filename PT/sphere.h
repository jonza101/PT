#pragma once

#include "shape.h"


class sphere : public shape
{
private:
	sphere(const float3 &pos, const float3 &orientation, float radius, material_data mat_data, float intensity)
	{
		this->type = SPHERE;
		this->is_light = intensity > 0.0f;

		this->pos = pos;
		this->orientation = orientation;
		this->radius = radius;

		this->intensity = intensity;

		this->albedo_id = mat_data.albedo_id;
		this->metalness_id = mat_data.metalness_id;
		this->reflectance_roughness_id = mat_data.reflectance_roughness_id;
		this->transparency_roughness_id = mat_data.transparency_roughness_id;
		this->reflectance_id = mat_data.reflectance_id;
		this->transparency_id = mat_data.transparency_id;
		this->absorption_id = mat_data.absorption_id;
		this->fresnel_reflectance_id = mat_data.fresnel_reflectance_id;
		this->ior_id = mat_data.ior_id;
		this->normal_id = mat_data.normal_id;
		this->emissive_id = mat_data.emissive_id;

		this->albedo = mat_data.albedo;
		this->metalness = mat_data.metalness;
		this->reflectance_roughness = mat_data.reflectance_roughness;
		this->transparency_roughness = mat_data.transparency_roughness;
		this->reflectance = mat_data.reflectance;
		this->transparency = mat_data.transparency;
		this->absorption = mat_data.absorption;
		this->fresnel_reflectance = mat_data.fresnel_reflectance;
		this->ior = mat_data.ior;
		this->emissive = mat_data.emissive;

		this->uv_scale = mat_data.uv_scale;
	}

public:
	float		radius;

	static void		create_sphere(const float3 &pos, const float3 &orientation, float radius, material_data mat_data, float intensity, std::vector<shape*> &obj, std::vector<vol_data> &vol)
	{
		obj.push_back(new sphere(pos, orientation, radius, mat_data, intensity));

		vol.push_back(vol_data());
		vol_data &_vol = vol[vol.size() - 1];
		_vol.shadow_visibility = mat_data.shadow_visibility;
		_vol.obj_type = SPHERE;
		_vol.range[0] = obj.size() - 1;
		_vol.range[1] = obj.size() - 1;

		_vol.vol_min = make_float3(pos.x - radius, pos.y - radius, pos.z - radius);
		_vol.vol_max = make_float3(pos.x + radius, pos.y + radius, pos.z + radius);
	}


	void			d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) override
	{
		h_scene->obj[id].type = this->type;
		h_scene->obj[id].is_light = this->is_light;

		h_scene->obj[id].pos = this->pos;
		h_scene->obj[id].orientation = this->orientation;
		h_scene->obj[id].radius = this->radius;

		h_scene->obj[id].albedo_id = this->albedo_id;
		h_scene->obj[id].metalness_id = this->metalness_id;
		h_scene->obj[id].reflectance_roughness_id = this->reflectance_roughness_id;
		h_scene->obj[id].transparency_roughness_id = this->transparency_roughness_id;
		h_scene->obj[id].reflectance_id = this->reflectance_id;
		h_scene->obj[id].transparency_id = this->transparency_id;
		h_scene->obj[id].absorption_id = this->absorption_id;
		h_scene->obj[id].fresnel_reflectance_id = this->fresnel_reflectance_id;
		h_scene->obj[id].ior_id = this->ior_id;
		h_scene->obj[id].normal_id = this->normal_id;
		h_scene->obj[id].emissive_id = this->emissive_id;

		h_scene->obj[id].albedo = this->albedo;
		h_scene->obj[id].metalness = this->metalness;
		h_scene->obj[id].reflectance_roughness = this->reflectance_roughness;
		h_scene->obj[id].transparency_roughness = this->transparency_roughness;
		h_scene->obj[id].reflectance = this->reflectance;
		h_scene->obj[id].transparency = this->transparency;
		h_scene->obj[id].absorption = this->absorption;
		h_scene->obj[id].fresnel_reflectance = this->fresnel_reflectance;
		h_scene->obj[id].ior = this->ior;
		h_scene->obj[id].emissive = this->emissive;

		h_scene->obj[id].uv_scale = this->uv_scale;
	}
};
