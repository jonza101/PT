#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <algorithm>

#include "vec_math.h"
#include "gpu_scene.h"


enum obj_type
{
	SPHERE,
	PLANE,
	TRIANGLE,
	SHAPE_TYPE_COUNT
};

class shape
{
public:
	int				type;
	int				is_light;

	float3			pos;
	float3			orientation;

	int				albedo_id = -1;
	int				metalness_id = -1;
	int				roughness_id = -1;
	int				normal_id = -1;
	int				emissive_id = -1;

	float3			albedo;
	float			metalness;
	float			roughness;
	float			emissive = 0.0f;

	float2			uv_scale;
	float			intensity;


	virtual void	d_malloc(gpu_scene *h_scene, const int id, cudaError_t &cuda_status) = 0;
};


struct					mesh_material_data
{
	int					albedo_id = -1;
	int					metalness_id = -1;
	int					roughness_id = -1;
	int					normal_id = -1;
	int					emissive_id = -1;

	float3				albedo;
	float				metalness;
	float				roughness;
	float				emissive;
};
