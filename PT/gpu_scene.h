#pragma once

#include <cuda_runtime.h>

#include "const.h"


struct				gpu_cam
{
	float3			pos;

	float3			forward;
	float3			up;
	float3			right;

	float			aspect_ratio;
	float			scale;

	float			z_near;
	float			z_far;
};


struct				d_light_data
{
	int				type;
	int				obj_id;

	float3			albedo;
	float			intensity;

	float3			dir;
	float			dir_factor;
};

struct				d_obj_data
{
	int				type;
	int				is_light;

	float3			pos;
	float3			orientation;

	float3			vert[3];
	float3			norm[3];
	float2			uv[3];

	int				albedo_id;
	int				metalness_id;
	int				reflectance_roughness_id;
	int				transparency_roughness_id;
	int				reflectance_id;
	int				transparency_id;
	int				absorption_id;
	int				fresnel_reflectance_id;
	int				ior_id;
	int				normal_id;
	int				emissive_id;

	float2			uv_scale;

	float3			albedo;
	float			metalness;
	float			reflectance_roughness;
	float			transparency_roughness;
	float			reflectance;
	float			transparency;
	float			absorption;
	float			fresnel_reflectance;
	float			ior;
	float			emissive;


	float			radius;
};

struct				d_vol_data
{
	int				obj_type;
	int				bvh_type;

	int				shadow_visibility;
	float3			bounds[2];
	int				id_range[2];

	float			plane_d_near[BOUNDING_PLANES];
	float			plane_d_far[BOUNDING_PLANES];
};

struct				gpu_scene
{
	int				obj_count;
	d_obj_data		*obj;

	int				vol_count;
	d_vol_data		*vol;
	float3			bvh_plane_normal[BOUNDING_PLANES];

	int				light_count;
	d_light_data	*light;

	float3			background_color;
	int				env_map_status;
	int				*env_map;

	float			env_ior;
};
