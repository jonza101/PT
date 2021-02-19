#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>
#include <limits.h>

#include "const.h"
#include "vec_math.h"
#include "gpu_scene.h"


enum obj_type
{
	SPHERE,
	TRIANGLE,
	MESH,
	SHAPE_TYPE_COUNT
};

class shape
{
public:
	int					type;
	int					is_light;

	float3				pos;
	float3				orientation;

	int					albedo_id;
	int					metalness_id;
	int					reflectance_roughness_id;
	int					transparency_roughness_id;
	int					reflectance_id;
	int					transparency_id;
	int					absorption_id;
	int					fresnel_reflectance_id;
	int					ior_id;
	int					normal_id;
	int					emissive_id;

	float3				albedo;
	float				metalness;
	float				reflectance_roughness;
	float				transparency_roughness;
	float				reflectance;
	float				transparency;
	float				absorption;
	float				fresnel_reflectance;
	float				ior;
	float				emissive;

	float2				uv_scale;
	float				intensity;


	virtual void		d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) = 0;
};


struct					material_data
{
	int					albedo_id = -1;
	int					metalness_id = -1;
	int					reflectance_roughness_id = -1;
	int					transparency_roughness_id = -1;
	int					reflectance_id = -1;
	int					transparency_id = -1;
	int					absorption_id = -1;
	int					fresnel_reflectance_id = -1;
	int					ior_id = -1;
	int					normal_id = -1;
	int					emissive_id = -1;

	float3				albedo = make_float3(1.0f, 1.0f, 1.0f);
	float				metalness = 1.0f;
	float				reflectance_roughness = 1.0f;
	float				transparency_roughness = 1.0f;
	float				reflectance = 1.0f;
	float				transparency = 0.0f;
	float				absorption = 0.0f;
	float				fresnel_reflectance = 0.0f;
	float				ior = 1.0f;
	float				emissive = 0.0f;

	float2				uv_scale = make_float2(1.0f, 1.0f);


	int					shadow_visibility = true;
};


enum bound_type
{
	BVH_BOX,
	BVH_PLANE,
	BVH_COUNT
};

#define D_BVH_D_NEAR { INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY }
#define D_BVH_D_FAR { -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY, -INFINITY }

struct					vol_data
{
	int					obj_type;
	int					bvh_type;

	int					shadow_visibility;
	int					range[2];

	float				plane_d_near[BOUNDING_PLANES] = D_BVH_D_NEAR;
	float				plane_d_far[BOUNDING_PLANES] = D_BVH_D_FAR;

	float3				vol_min = make_float3(INFINITY, INFINITY, INFINITY);
	float3				vol_max = make_float3(-INFINITY, -INFINITY, -INFINITY);
};
