﻿
#include "pt.h"


//	MATH
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ float	clamp(float val, float min_val, float max_val)
{
	return (fminf(max_val, fmaxf(min_val, val)));
}

__device__ float	length(const float3 &a)
{
	return (sqrtf(a.x * a.x + a.y * a.y + a.z * a.z));
}

__device__ float	dot(const float3 &a, const float3 &b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float3	cross(const float3 &a, const float3 &b)
{
	float3 vec;

	vec.x = a.y * b.z - a.z * b.y;
	vec.y = a.z * b.x - a.x * b.z;
	vec.z = a.x * b.y - a.y * b.x;

	return (vec);
}

__device__ float3	normalize(const float3 &a)
{
	float3 vec;

	float len = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
	vec.x = (float)a.x / (float)len;
	vec.y = (float)a.y / (float)len;
	vec.z = (float)a.z / (float)len;

	return (vec);

}

__device__ float	lerp(float a, float b, float factor)
{
	float v = a + factor * (b - a);

	return (v);
}

__device__ float3	lerp(const float3 &a, const float3 &b, float factor)
{
	float3 vec;

	vec.x = a.x + factor * (b.x - a.x);
	vec.y = a.y + factor * (b.y - a.y);
	vec.z = a.z + factor * (b.z - a.z);

	return (vec);
}


__device__ float3	reflect(const float3 &i, const float3 &n)
{
	float3 r;

	float n_dot_i = dot(n, i);
	r.x = i.x - 2.0f * n.x * n_dot_i;
	r.y = i.y - 2.0f * n.y * n_dot_i;
	r.z = i.z - 2.0f * n.z * n_dot_i;

	return (r);
}

__device__ float3	refract(const float3 &i, const float3 &n, float etai, float etat)
{
	float cosi = clamp(-1.0f, 1.0f, dot(i, n));
	if (cosi < 0.0f)
		cosi = -cosi;
	float eta = (float)etai / (float)etat;
	float k = 1.0f - eta * eta * (1.0f - cosi * cosi); 
	if (k < 0.0f)
		return (make_float3(0.0f, 0.0f, 0.0f));				//	i

	k = sqrtf(k);
	float3 r;
	r.x = eta * i.x + (eta * cosi - k) * n.x;
	r.y = eta * i.y + (eta * cosi - k) * n.y;
	r.z = eta * i.z + (eta * cosi - k) * n.z;

	return (r);
}

__device__ float	fresnel(const float3 &i, const float3 &n, float etai, float etat)
{
	float r0 = (float)(etai - etat) / (float)(etai + etat);
	r0 *= r0;
	float cosx = -dot(i, n);
	if (etai > etat)
	{
		float eta = (float)etai / (float)etat;
		float sint2 = eta * eta * (1.0f - cosx * cosx);
		if (sint2 > 1.0f)
			return (1.0f);
		cosx = sqrtf(1.0f - sint2);
	}
	float x = 1.0f - cosx;
	float ret = r0 + (1.0f - r0) * x * x * x * x;

	return (ret);
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__device__ float3	operator * (const float3 &lhs, const float3 &rhs)
{
	float3 vec;
	vec.x = lhs.x * rhs.x;
	vec.y = lhs.y * rhs.y;
	vec.z = lhs.z * rhs.z;

	return (vec);
}


__device__ float3	sample_brdf(const float3 &normal, curandState *curand_state, float roughness)
{
	float r1 = curand_uniform(*&curand_state);
	float r2 = curand_uniform(*&curand_state);

	float theta = atanf(roughness * sqrtf(1.0f - r1 * r1));
	float phi = 2.0f * M_PI * r2;

	float x = theta * cosf(phi);
	float y = r1;
	float z = theta * sinf(phi);

	float3 t, b;
	t.x = 0.0f;
	t.y = -normal.z;
	t.z = normal.y;
	if (fabs(normal.x) > fabs(normal.y))
	{
		t.x = normal.z;
		t.y = 0.0f;
		t.z = -normal.x;
	}

	t = normalize(t);
	b = cross(normal, t);

	float3 rand_dir;
	rand_dir.x = x * b.x + y * normal.x + z * t.x;
	rand_dir.y = x * b.y + y * normal.y + z * t.y;
	rand_dir.z = x * b.z + y * normal.z + z * t.z;
	rand_dir = normalize(rand_dir);

	return (rand_dir);
}


__device__ float3	vec_rotate(const float3 &v, const float3 &k, float theta)
{
	float cost = cosf(theta);
	float sint = sinf(theta);
	float3 c = cross(k, v);
	float k_dot_v = dot(k, v);

	float3 r;
	r.x = (v.x * cost) + (c.x * sint) + (k.x * k_dot_v) * (1.0f - cost);
	r.y = (v.y * cost) + (c.y * sint) + (k.y * k_dot_v) * (1.0f - cost);
	r.z = (v.z * cost) + (c.z * sint) + (k.z * k_dot_v) * (1.0f - cost);

	return (r);
}


__device__ float3	tex_2d(const float2 &uv, gpu_tex *g_tex, int tex_id)
{
	float3 color;

	int tx = uv.x * g_tex->wh[tex_id].x;
	int ty = uv.y * g_tex->wh[tex_id].y;
	int hex_color = g_tex->data[g_tex->offset[tex_id] + ty * g_tex->wh[tex_id].x + tx];

	color.x = (float)((hex_color >> 16) & 0xFF) / (float)255.0f;
	color.y = (float)((hex_color >> 8) & 0xFF) / (float)255.0f;
	color.z = (float)(hex_color & 0xFF) / (float)255.0f;

	return (color);
}

__device__ float3	tex_cube(const float3 &d, gpu_tex *g_tex, gpu_scene *scene)
{
	float3 color;

	float abs_x = fabsf(d.x);
	float abs_y = fabsf(d.y);
	float abs_z = fabsf(d.z);
	float sc, tc, ma;

	int idx;
	float2 uv;

	if (abs_x >= abs_y && abs_x >= abs_z)
	{
		if (d.x > 0.0f)
		{
			idx = 0;
			sc = -d.z;
			tc = -d.y;
			ma = abs_x;

		}
		else
		{
			idx = 1;
			sc = d.z;
			tc = -d.y;
			ma = abs_x;

		}
	}
	if (abs_y >= abs_x && abs_y >= abs_z)
	{
		if (d.y > 0.0f)
		{
			idx = 2;
			sc = d.x;
			tc = d.z;
			ma = abs_y;

		}
		else
		{
			idx = 3;
			sc = d.x;
			tc = -d.z;
			ma = abs_y;
		}
	}
	if (abs_z >= abs_x && abs_z >= abs_y)
	{
		if (d.z > 0.0f)
		{
			idx = 4;
			sc = d.x;
			tc = -d.y;
			ma = abs_z;

		}
		else
		{
			idx = 5;
			sc = -d.x;
			tc = -d.y;
			ma = abs_z;
		}
	}

	if (ma == 0.0f)
	{
		uv.x = 0.0f;
		uv.y = 0.0f;
	}
	else
	{
		uv.x = ((float)sc / (float)ma + 1.0f) * 0.5f;
		uv.y = ((float)tc / (float)ma + 1.0f) * 0.5f;
	}

	return (tex_2d(uv, g_tex, scene->env_map[idx]));
}


__device__ int		box_volume_intersection(const float3 &origin, const float3 &dir, gpu_scene *scene, int vol_id)
{
	float tx_min, tx_max;
	float ty_min, ty_max;
	float tz_min, tz_max;

	float3 inv_dir;
	inv_dir.x = 1.0f / (float)dir.x;
	inv_dir.y = 1.0f / (float)dir.y;
	inv_dir.z = 1.0f / (float)dir.z;
	int sign[3] = { inv_dir.x < 0.0f, inv_dir.y < 0.0f, inv_dir.z < 0.0f };

	tx_min = (scene->vol[vol_id].bounds[sign[0]].x - origin.x) * inv_dir.x;
	tx_max = (scene->vol[vol_id].bounds[1 - sign[0]].x - origin.x) * inv_dir.x;
	ty_min = (scene->vol[vol_id].bounds[sign[1]].y - origin.y) * inv_dir.y;
	ty_max = (scene->vol[vol_id].bounds[1 - sign[1]].y - origin.y) * inv_dir.y;
	if (tx_min > ty_max || ty_min > tx_max)
		return (0);
	if (ty_min > tx_min)
		tx_min = ty_min;
	if (ty_max < tx_max)
		tx_max = ty_max;

	tz_min = (scene->vol[vol_id].bounds[sign[2]].z - origin.z) * inv_dir.z;
	tz_max = (scene->vol[vol_id].bounds[1 - sign[2]].z - origin.z) * inv_dir.z;
	if (tx_min > tz_max || tz_min > tx_max)
		return (0);
	/*if (tz_min > tx_min)
		tx_min = tz_min; 
	if (tz_max < tx_max)
		tx_max = tz_max;*/

	return (1);
}

__device__ int		plane_volume_intersection(const float3 &origin, const float3 &dir, gpu_scene *scene, int vol_id, float *n_dot_o, float *n_dot_d)
{
	float t_near = -INFINITY;
	float t_far = INFINITY;
	
	int i = -1;
	while (++i < BOUNDING_PLANES)
	{
		float near = (float)(scene->vol[vol_id].plane_d_near[i] - n_dot_o[i]) / (float)n_dot_d[i];
		float far = (float)(scene->vol[vol_id].plane_d_far[i] - n_dot_o[i]) / (float)n_dot_d[i];
		if (n_dot_d[i] < 0.0f)
		{
			float temp = near;
			near = far;
			far = temp;
		}
		if (near > t_near)
			t_near = near;
		if (far < t_far)
			t_far = far;
	
		if (t_near > t_far)
			return (0);
	}

	return (1);
}


__device__ int		shadow_factor(const float3 &origin, const float3 &dir, float z_near, float z_far, gpu_scene *scene, int light_id)
{
	float3 oc;

	float bvh_n_dot_o[BOUNDING_PLANES];
	float bvh_n_dot_d[BOUNDING_PLANES];
	int b = -1;
	while (++b < BOUNDING_PLANES)
	{
		bvh_n_dot_o[b] = dot(scene->bvh_plane_normal[b], origin);
		bvh_n_dot_d[b] = dot(scene->bvh_plane_normal[b], dir);
	}

	//float3 inv_dir;
	//inv_dir.x = 1.0f / (float)dir.x;
	//inv_dir.y = 1.0f / (float)dir.y;
	//inv_dir.z = 1.0f / (float)dir.z;
	//
	//int sign[3];
	//sign[0] = inv_dir.x < 0.0f;
	//sign[1] = inv_dir.y < 0.0f;
	//sign[2] = inv_dir.z < 0.0f;


	int v = -1;
	while (++v < scene->vol_count)
	{
		if (!scene->vol[v].shadow_visibility)
			continue;

		switch (scene->vol[v].obj_type)
		{
			case SPHERE:
			{
				int id = scene->vol[v].id_range[0];
				float t = -1.0f;

				oc.x = scene->obj[id].pos.x - origin.x;
				oc.y = scene->obj[id].pos.y - origin.y;
				oc.z = scene->obj[id].pos.z - origin.z;

				float tca = dot(oc, dir);
				if (tca < 0.0f)
					break;

				float radius2 = scene->obj[id].radius * scene->obj[id].radius;
				float d2 = dot(oc, oc) - tca * tca;
				if (d2 > radius2)
					break;

				float thc = sqrtf(radius2 - d2);
				float t0 = tca - thc;
				float t1 = tca + thc;
				if (t0 > t1)
				{
					float temp = t0;
					t0 = t1;
					t1 = temp;
				}
				if (t0 < 0.0f)
				{
					t0 = t1;
					if (t0 < 0.0f)
						break;
				}
				t = t0;
				if (t >= z_near && t <= z_far)
					return (0);

				break;
			}
			case MESH:
			{
				int bvh_type = scene->vol[v].bvh_type;
				int intersect = 0;
				switch (bvh_type)
				{
					case BVH_BOX:
					{
						intersect = box_volume_intersection(origin, dir, scene, v);
						break;
					}
					case BVH_PLANE:
					{
						intersect = plane_volume_intersection(origin, dir, scene, v, bvh_n_dot_o, bvh_n_dot_d);
						break;
					}
				}
				if (!intersect)
					continue;
				

				int i = scene->vol[v].id_range[0] - 1;
				while (++i <= scene->vol[v].id_range[1])
				{
					float t = -1.0f;

					float3 a = scene->obj[i].vert[0];
					float3 b = scene->obj[i].vert[1];
					float3 c = scene->obj[i].vert[2];

					float3 e1, e2;
					float3 tvec, qvec;

					e1.x = b.x - a.x;
					e1.y = b.y - a.y;
					e1.z = b.z - a.z;
					e2.x = c.x - a.x;
					e2.y = c.y - a.y;
					e2.z = c.z - a.z;

					float3 pvec = cross(dir, e2);
					float det = dot(e1, pvec);

					if (det < EPSILON && det > -EPSILON)
						continue;

					float inv_det = 1.0f / (float)det;
					tvec.x = origin.x - a.x;
					tvec.y = origin.y - a.y;
					tvec.z = origin.z - a.z;
					float u = dot(tvec, pvec) * inv_det;
					if (u < 0.0f || u > 1.0f)
						continue;

					qvec = cross(tvec, e1);
					float v = dot(dir, qvec) * inv_det;
					if (v < 0.0f || u + v > 1.0f)
						continue;

					t = dot(e2, qvec) * inv_det;
					if (t >= z_near && t <= z_far)
						return (0);
				}

				break;
			}
		}
	}

	return (1);
}

__device__ int		intersection(const float3 &origin, const float3 &dir, float z_near, float z_far, gpu_scene *scene, float &dist)
{
	dist = z_far;
	int obj_id = -1;
	float3 oc;
	
	float bvh_n_dot_o[BOUNDING_PLANES];
	float bvh_n_dot_d[BOUNDING_PLANES];
	int b = -1;
	while (++b < BOUNDING_PLANES)
	{
		bvh_n_dot_o[b] = dot(scene->bvh_plane_normal[b], origin);
		bvh_n_dot_d[b] = dot(scene->bvh_plane_normal[b], dir);
	}

	//float3 inv_dir;
	//inv_dir.x = 1.0f / (float)dir.x;
	//inv_dir.y = 1.0f / (float)dir.y;
	//inv_dir.z = 1.0f / (float)dir.z;
	//
	//int sign[3];
	//sign[0] = inv_dir.x < 0.0f;
	//sign[1] = inv_dir.y < 0.0f;
	//sign[2] = inv_dir.z < 0.0f;


	int v = -1;
	while (++v < scene->vol_count)
	{
		switch (scene->vol[v].obj_type)
		{
			case SPHERE:
			{
				int id = scene->vol[v].id_range[0];
				float t = -1.0f;

				oc.x = scene->obj[id].pos.x - origin.x;
				oc.y = scene->obj[id].pos.y - origin.y;
				oc.z = scene->obj[id].pos.z - origin.z;

				float tca = dot(oc, dir);
				if (tca < 0.0f)
					break;

				float radius2 = scene->obj[id].radius * scene->obj[id].radius;
				float d2 = dot(oc, oc) - tca * tca;
				if (d2 > radius2)
					break;

				float thc = sqrtf(radius2 - d2);
				float t0 = tca - thc;
				float t1 = tca + thc;
				if (t0 > t1)
				{
					float temp = t0;
					t0 = t1;
					t1 = temp;
				}
				if (t0 < 0.0f)
				{
					t0 = t1;
					if (t0 < 0.0f)
						break;
				}
				t = t0;
				if (t >= z_near && t <= z_far && t < dist)
				{
					obj_id = id;
					dist = t;
				}

				break;
			}
			case MESH:
			{
				int bvh_type = scene->vol[v].bvh_type;
				int intersect = 0;
				switch (bvh_type)
				{
					case BVH_BOX:
					{
						intersect = box_volume_intersection(origin, dir, scene, v);
						break;
					}
					case BVH_PLANE:
					{
						intersect = plane_volume_intersection(origin, dir, scene, v, bvh_n_dot_o, bvh_n_dot_d);
						break;
					}
				}
				if (!intersect)
					continue;
			

				int i = scene->vol[v].id_range[0] - 1;
				while (++i <= scene->vol[v].id_range[1])
				{
					float t = -1.0f;

					float3 a = scene->obj[i].vert[0];
					float3 b = scene->obj[i].vert[1];
					float3 c = scene->obj[i].vert[2];

					float3 e1, e2;
					float3 tvec, qvec;

					e1.x = b.x - a.x;
					e1.y = b.y - a.y;
					e1.z = b.z - a.z;
					e2.x = c.x - a.x;
					e2.y = c.y - a.y;
					e2.z = c.z - a.z;

					float3 pvec = cross(dir, e2);
					float det = dot(e1, pvec);

					if (det < EPSILON && det > -EPSILON)
						continue;

					float inv_det = 1.0f / (float)det;
					tvec.x = origin.x - a.x;
					tvec.y = origin.y - a.y;
					tvec.z = origin.z - a.z;
					float u = dot(tvec, pvec) * inv_det;
					if (u < 0.0f || u > 1.0f)
						continue;

					qvec = cross(tvec, e1);
					float v = dot(dir, qvec) * inv_det;
					if (v < 0.0f || u + v > 1.0f)
						continue;

					t = dot(e2, qvec) * inv_det;
					if (t >= z_near && t <= z_far && t < dist)
					{
						obj_id = i;
						dist = t;
					}
				}

				break;
			}
		}
	}

	return (obj_id);
}


__device__ float2	generate_uv(const float3 &point, const float3 &normal, const float3 &barycentric, gpu_scene *scene, gpu_tex *g_tex, int obj_id)
{
	float2 uv = make_float2(0.0f, 0.0f);
	switch (scene->obj[obj_id].type)
	{
		case SPHERE:
		{
			float3 c = scene->obj[obj_id].pos;
			float3 orientation = scene->obj[obj_id].orientation;
			float radius = scene->obj[obj_id].radius;
			float2 uv_scale = scene->obj[obj_id].uv_scale;


			float3 r_point;
			float3 r_normal;

			r_point.x = point.x - c.x;
			r_point.y = point.y - c.y;
			r_point.z = point.z - c.z;

			r_point = vec_rotate(r_point, make_float3(1.0f, 0.0f, 0.0f), orientation.x);
			r_point = vec_rotate(r_point, make_float3(0.0f, 1.0f, 0.0f), orientation.y);
			r_point = vec_rotate(r_point, make_float3(0.0f, 0.0f, 1.0f), orientation.z);

			r_point.x += c.x;
			r_point.y += c.y;
			r_point.z += c.z;

			r_normal.x = (float)(r_point.x - c.x) / (float)radius;
			r_normal.y = (float)(r_point.y - c.y) / (float)radius;
			r_normal.z = (float)(r_point.z - c.z) / (float)radius;


			uv.x = fmodf(fabsf((0.5f + (float)atan2f(r_normal.z, r_normal.x) / (float)(2.0f * M_PI)) * uv_scale.x), 1.0f);
			uv.y = fmodf(fabsf((0.5f - (float)asinf(r_normal.y) / (float)M_PI) * uv_scale.y), 1.0f);

			break;
		}
		case TRIANGLE:
		{
			float2 uv_scale = scene->obj[obj_id].uv_scale;

			float2 a_uv = scene->obj[obj_id].uv[0];
			float2 b_uv = scene->obj[obj_id].uv[1];
			float2 c_uv = scene->obj[obj_id].uv[2];

			uv.x = barycentric.x * a_uv.x + barycentric.y * b_uv.x + barycentric.z * c_uv.x;
			uv.y = barycentric.x * a_uv.y + barycentric.y * b_uv.y + barycentric.z * c_uv.y;

			//uv.x = fminf(1.0f, fmaxf(0.0f, uv.x));
			//uv.y = fminf(1.0f, fmaxf(0.0f, uv.y));
			uv.x = clamp(fmodf(uv.x * uv_scale.x, 1.0f), 0.0f, 1.0f);
			uv.y = clamp(fmodf(uv.y * uv_scale.y, 1.0f), 0.0f, 1.0f);

			break;
		}
	}

	return (uv);
}


__device__ float3	normal_map(const float3 &normal, const float2 &uv, gpu_tex *g_tex, int tex_id)
{
	float3 new_n;
	float3 n = normal;

	float3 t = cross(normal, make_float3(0.0f, 1.0f, 0.0f));
	if (!length(t))
		t = cross(normal, make_float3(0.0f, 0.0f, 1.0f));
	t = normalize(t);
	float3 b = normalize(cross(normal, t));

	float3 map_n = tex_2d(uv, g_tex, tex_id);
	map_n.x = map_n.x * 2.0f - 1.0f;
	map_n.y = map_n.y * 2.0f - 1.0f;
	map_n.z = map_n.z * 2.0f - 1.0f;

	new_n.x = t.x * map_n.x + b.x * map_n.y + n.x * map_n.z;
	new_n.y = t.y * map_n.x + b.y * map_n.y + n.y * map_n.z;
	new_n.z = t.z * map_n.x + b.z * map_n.y + n.z * map_n.z;
	new_n = normalize(new_n);

	return (new_n);
}



__device__ float	distr_trowbridge_reitz_ggx(const float3 &normal, const float3 &halfway_dir, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float n_dot_h = fmaxf(0.0f, dot(normal, halfway_dir));
	float n_dot_h2 = n_dot_h * n_dot_h;

	float num = a2;
	float den = n_dot_h2 * (a2 - 1.0f) + 1.0f;
	den = M_PI * den * den;

	return ((float)num / (float)den);
}

__device__ float	geom_schlick_ggx(float n_dot_v, float roughness)
{
	float r = (roughness + 1.0f);
	float k = (float)(r * r) / 8.0f;

	float num = n_dot_v;
	float den = n_dot_v * (1.0f - k) + k;

	return ((float)num / (float)den);
}

__device__ float	geom_smith(const float3 &normal, const float3 &view_dir, const float3 &light_dir, float roughness)
{
	float n_dot_v = fmaxf(0.0f, dot(normal, view_dir));
	float n_dot_l = fmaxf(0.0f, dot(normal, light_dir));

	float ggx1 = geom_schlick_ggx(n_dot_v, roughness);
	float ggx2 = geom_schlick_ggx(n_dot_l, roughness);

	return (ggx1 * ggx2);
}

__device__ float3	fresnel_schlick(float cost, const float3 &f0)
{
	float3 f;

	float pw = powf(1.0f - cost, 5.0f);
	f.x = f0.x + (1.0f - f0.x) * pw;
	f.y = f0.y + (1.0f - f0.y) * pw;
	f.z = f0.z + (1.0f - f0.z) * pw;

	return (f);
}

__device__ float3	estimate_direct(float3 &point, float3 &normal, gpu_scene *scene, gpu_tex *g_tex, curandState *curand_state, int hit_obj_id, const float3 &view_dir, const float3 &albedo, float roughness, float metalness)
{
	float3 lo;
	lo.x = 0.0f;
	lo.y = 0.0f;
	lo.z = 0.0f;

	float3 f0;
	f0.x = 0.01f;
	f0.y = 0.01f;
	f0.z = 0.01f;
	f0 = lerp(f0, albedo, metalness);


	float3 light_dir, halfway_dir, light_pos;
	float3 sample_point, sample_dir;

	int l_id;
	float l_radius, l_dist, l_area;
	float attenuation;
	float3 emission;
	float3 radiance;

	int l = -1;
	while (++l < scene->light_count)
	{
		int l_type = scene->light[l].type;

		l_area = 1.0f;
		attenuation = 1.0f;

		switch (l_type)
		{
			case SPHERICAL:
			{
				l_id = scene->light[l].obj_id;
				l_radius = scene->obj[l_id].radius;
				if (l_id == hit_obj_id)
					continue;

				light_pos = scene->obj[l_id].pos;

				light_dir.x = light_pos.x - point.x;
				light_dir.y = light_pos.y - point.y;
				light_dir.z = light_pos.z - point.z;
				l_dist = sqrtf(light_dir.x * light_dir.x + light_dir.y * light_dir.y + light_dir.z * light_dir.z) - l_radius;
				l_dist = fmaxf(l_dist, EPSILON);
				float l_dist_sqrt = sqrtf(l_dist);
				light_dir = normalize(light_dir);

				l_area = 4.0f * M_PI * l_dist_sqrt;
				l_area = fmaxf(l_area, EPSILON);
				attenuation = 1.0f / (float)(l_dist * l_dist);
				emission = scene->obj[l_id].albedo;

				break;
			}
			case DIRECTIONAL:
			{
				float3 l_dir = scene->light[l].dir;
				light_dir.x = -l_dir.x;
				light_dir.y = -l_dir.y;
				light_dir.z = -l_dir.z;
				l_dist = FLT_MAX;

				emission = scene->light[l].albedo;

				break;
			}
		}

		float r1 = curand_uniform(*&curand_state);
		float r2 = curand_uniform(*&curand_state);
		switch (l_type)
		{
			case SPHERICAL:
			{
				float3 point_dir;
				point_dir.x = -light_dir.x;
				point_dir.y = -light_dir.y;
				point_dir.z = -light_dir.z;

				float3 hemi_dir = sample_brdf(point_dir, curand_state, 1.0f);
				sample_point.x = light_pos.x + hemi_dir.x * l_radius;
				sample_point.y = light_pos.y + hemi_dir.y * l_radius;
				sample_point.z = light_pos.z + hemi_dir.z * l_radius;

				sample_dir.x = sample_point.x - point.x;
				sample_dir.y = sample_point.y - point.y;
				sample_dir.z = sample_point.z - point.z;
				sample_dir = normalize(sample_dir);


				if (scene->obj[l_id].albedo_id >= 0)
				{
					float2 uv = generate_uv(sample_point, hemi_dir, make_float3(0.0f, 0.0f, 0.0f), scene, g_tex, hit_obj_id);
					emission = tex_2d(uv, g_tex, scene->obj[l_id].albedo_id);
				}

				break;
			}
			case DIRECTIONAL:
			{
				float dir_factor = scene->light[l].dir_factor;

				float theta = atanf(dir_factor * sqrtf(1.0f - r1 * r1));
				float phi = 2.0f * M_PI * r2;

				float x = theta * cosf(phi);
				float y = r1;
				float z = theta * sinf(phi);

				float3 t, b;
				t.x = 0.0f;
				t.y = -light_dir.z;
				t.z = light_dir.y;
				if (fabs(light_dir.x) > fabs(light_dir.y))
				{
					t.x = light_dir.z;
					t.y = 0.0f;
					t.z = -light_dir.x;
				}

				t = normalize(t);
				b = cross(light_dir, t);

				float3 rand_dir;
				rand_dir.x = x * b.x + y * light_dir.x + z * t.x;
				rand_dir.y = x * b.y + y * light_dir.y + z * t.y;
				rand_dir.z = x * b.z + y * light_dir.z + z * t.z;
				rand_dir = normalize(rand_dir);

				sample_dir = lerp(light_dir, rand_dir, dir_factor);
				sample_dir = normalize(sample_dir);

				break;
			}
		}

		float sh_hit = shadow_factor(point, sample_dir, RAY_OFFSET, l_dist, scene, l);
		float sh_factor = (float)sh_hit;


		radiance.x = emission.x * attenuation / (float)l_area * scene->light[l].intensity;
		radiance.y = emission.y * attenuation / (float)l_area * scene->light[l].intensity;
		radiance.z = emission.z * attenuation / (float)l_area * scene->light[l].intensity;


		halfway_dir.x = view_dir.x + light_dir.x;
		halfway_dir.y = view_dir.y + light_dir.y;
		halfway_dir.z = view_dir.z + light_dir.z;
		halfway_dir = normalize(halfway_dir);
		float cost = fmaxf(0.0f, dot(normal, light_dir));

		float d = distr_trowbridge_reitz_ggx(normal, halfway_dir, roughness);
		float g = geom_smith(normal, view_dir, light_dir, roughness);
		float3 f = fresnel_schlick(fmaxf(0.0f, dot(halfway_dir, view_dir)), f0);

		float3 ks = f;
		float3 kd;
		kd.x = (1.0f - ks.x) * (1.0f - metalness);
		kd.y = (1.0f - ks.y) * (1.0f - metalness);
		kd.z = (1.0f - ks.z) * (1.0f - metalness);

		float3 num;
		num.x = d * g * f.x;
		num.y = d * g * f.y;
		num.z = d * g * f.z;
		float den = 4.0f * fmaxf(0.0f, dot(normal, view_dir)) * cost;
		den = fmaxf(0.001f, den);

		float3 specular;
		specular.x = (float)num.x / (float)den;
		specular.y = (float)num.y / (float)den;
		specular.z = (float)num.z / (float)den;

		lo.x += ((float)kd.x * (float)albedo.x / (float)M_PI + specular.x) * radiance.x * cost * sh_factor;
		lo.y += ((float)kd.y * (float)albedo.y / (float)M_PI + specular.y) * radiance.y * cost * sh_factor;
		lo.z += ((float)kd.z * (float)albedo.z / (float)M_PI + specular.z) * radiance.z * cost * sh_factor;
	}

	lo.x = fminf(1.0f, fmaxf(0.0f, lo.x));
	lo.y = fminf(1.0f, fmaxf(0.0f, lo.y));
	lo.z = fminf(1.0f, fmaxf(0.0f, lo.z));

	return (lo);
}



__device__ float3	trace(float3 &origin, float3 &dir, float z_near, float z_far, gpu_scene *scene, gpu_tex *g_tex, curandState *curand_state)
{
	float3 curr_origin = origin;
	float3 curr_dir = dir;
	float3 view_dir;
	
	float env_ior = scene->env_ior;


	float3 color;
	color.x = 0.0f;
	color.y = 0.0f;
	color.z = 0.0f;

	float3 throughput;
	throughput.x = 1.0f;
	throughput.y = 1.0f;
	throughput.z = 1.0f;

	float3 point;
	float3 normal;
	float3 barycentric;


	int b = -1;
	while (++b < MAX_BOUNCES)
	{
		float dist;
		int inside_factor = 1;
		int hit_obj_id = intersection(curr_origin, curr_dir, z_near, z_far, scene, dist);

		if (hit_obj_id >= 0)
		{
			point.x = curr_origin.x + curr_dir.x * dist;
			point.y = curr_origin.y + curr_dir.y * dist;
			point.z = curr_origin.z + curr_dir.z * dist;

			view_dir.x = -curr_dir.x;
			view_dir.y = -curr_dir.y;
			view_dir.z = -curr_dir.z;


			switch (scene->obj[hit_obj_id].type)
			{
				case SPHERE:
				{
					float3 oc;
					oc.x = scene->obj[hit_obj_id].pos.x - curr_origin.x;
					oc.y = scene->obj[hit_obj_id].pos.y - curr_origin.y;
					oc.z = scene->obj[hit_obj_id].pos.z - curr_origin.z;
					float tca = dot(oc, curr_dir);
					float radius2 = scene->obj[hit_obj_id].radius * scene->obj[hit_obj_id].radius;
					float d2 = dot(oc, oc) - tca * tca;
					float thc = sqrtf(radius2 - d2);
					float t0 = tca - thc;
					float t1 = tca + thc;
					if (t0 > t1)
					{
						float temp = t0;
						t0 = t1;
						t1 = temp;
					}
					if (t0 < 0.0f)
					{
						inside_factor = -1;
					}

					normal.x = (float)(point.x - scene->obj[hit_obj_id].pos.x) / (float)scene->obj[hit_obj_id].radius;
					normal.y = (float)(point.y - scene->obj[hit_obj_id].pos.y) / (float)scene->obj[hit_obj_id].radius;
					normal.z = (float)(point.z - scene->obj[hit_obj_id].pos.z) / (float)scene->obj[hit_obj_id].radius;
					normal = normalize(normal);

					normal.x *= inside_factor;
					normal.y *= inside_factor;
					normal.z *= inside_factor;

					break;
				}
				case TRIANGLE:
				{
					float3 a = scene->obj[hit_obj_id].vert[0];
					float3 b = scene->obj[hit_obj_id].vert[1];
					float3 c = scene->obj[hit_obj_id].vert[2];

					float3 a_n = scene->obj[hit_obj_id].norm[0];
					float3 b_n = scene->obj[hit_obj_id].norm[1];
					float3 c_n = scene->obj[hit_obj_id].norm[2];

					float3 v0, v1, v2;
					v0.x = b.x - a.x;
					v0.y = b.y - a.y;
					v0.z = b.z - a.z;
					v1.x = c.x - a.x;
					v1.y = c.y - a.y;
					v1.z = c.z - a.z;
					v2.x = point.x - a.x;
					v2.y = point.y - a.y;
					v2.z = point.z - a.z;

					float det = dot(v0, cross(curr_dir, v1));
					if (det < 0.0f)
						inside_factor = -1;

					float d00 = dot(v0, v0);
					float d01 = dot(v0, v1);
					float d11 = dot(v1, v1);
					float d20 = dot(v2, v0);
					float d21 = dot(v2, v1);
					float denom = d00 * d11 - d01 * d01;

					barycentric.y = (float)(d11 * d20 - d01 * d21) / (float)denom;
					barycentric.z = (float)(d00 * d21 - d01 * d20) / (float)denom;
					barycentric.x = 1.0f - barycentric.y - barycentric.z;

					normal.x = barycentric.x * a_n.x + barycentric.y * b_n.x + barycentric.z * c_n.x;
					normal.y = barycentric.x * a_n.y + barycentric.y * b_n.y + barycentric.z * c_n.y;
					normal.z = barycentric.x * a_n.z + barycentric.y * b_n.z + barycentric.z * c_n.z;
					normal = normalize(normal);

					normal.x *= inside_factor;
					normal.y *= inside_factor;
					normal.z *= inside_factor;

					break;
				}
			}

			float2 uv;
			float3 albedo = scene->obj[hit_obj_id].albedo;
			float metalness = scene->obj[hit_obj_id].metalness;
			float reflectance_roughness = scene->obj[hit_obj_id].reflectance_roughness;
			float transparency_roughness = scene->obj[hit_obj_id].transparency_roughness;
			float reflectance = scene->obj[hit_obj_id].reflectance;
			float transparency = scene->obj[hit_obj_id].transparency;
			float absorption = scene->obj[hit_obj_id].absorption;
			float fresnel_reflectance = scene->obj[hit_obj_id].fresnel_reflectance;
			float ior = scene->obj[hit_obj_id].ior;
			float emissive = scene->obj[hit_obj_id].emissive;

			int albedo_id = scene->obj[hit_obj_id].albedo_id;
			int metalness_id = scene->obj[hit_obj_id].metalness_id;
			int reflectance_roughness_id = scene->obj[hit_obj_id].reflectance_roughness_id;
			int transparency_roughness_id = scene->obj[hit_obj_id].transparency_roughness_id;
			int reflectance_id = scene->obj[hit_obj_id].reflectance_id;
			int transparency_id = scene->obj[hit_obj_id].transparency_id;
			int absorption_id = scene->obj[hit_obj_id].absorption_id;
			int fresnel_reflectance_id = scene->obj[hit_obj_id].fresnel_reflectance_id;
			int ior_id = scene->obj[hit_obj_id].ior_id;
			int normal_id = scene->obj[hit_obj_id].normal_id;
			int emissive_id = scene->obj[hit_obj_id].emissive_id;

			if (albedo_id >= 0 || metalness_id >= 0 || reflectance_roughness_id >= 0 || transparency_roughness_id >= 0
								|| normal_id >= 0 || emissive_id >= 0 || reflectance_id >= 0 || absorption_id >= 0
								|| fresnel_reflectance_id >= 0 || transparency_id >= 0 || ior_id >= 0)
			{
				uv = generate_uv(point, normal, barycentric, scene, g_tex, hit_obj_id);
			}

			if (albedo_id >= 0)
				albedo = tex_2d(uv, g_tex, albedo_id) * albedo;
			if (metalness_id >= 0)
				metalness = tex_2d(uv, g_tex, metalness_id).x * metalness;
			if (reflectance_roughness_id >= 0)
				reflectance_roughness = tex_2d(uv, g_tex, reflectance_roughness_id).x * reflectance_roughness;
			if (transparency_roughness_id >= 0)
				transparency_roughness = tex_2d(uv, g_tex, transparency_roughness_id).x * transparency_roughness;
			if (reflectance_id >= 0)
				reflectance = tex_2d(uv, g_tex, reflectance_id).x;
			if (transparency_id >= 0)
				transparency = tex_2d(uv, g_tex, reflectance_id).x;
			if (absorption_id >= 0)
				absorption = tex_2d(uv, g_tex, absorption_id).x * absorption;
			if (fresnel_reflectance_id >= 0)
				fresnel_reflectance = tex_2d(uv, g_tex, fresnel_reflectance_id).x * fresnel_reflectance;
			if (ior_id >= 0)
				ior = tex_2d(uv, g_tex, ior_id).x * ior;
			if (normal_id >= 0)
				normal = normal_map(normal, uv, g_tex, normal_id);
			if (emissive_id >= 0)
				emissive = tex_2d(uv, g_tex, emissive_id).x * emissive;


			float3 d_brdf, s_brdf, brdf;
			float pdf = 1.0f, cost = 0.0f;
			float roughness = reflectance_roughness;

			float3 perfect_ray, hemi_sample_ray;
			perfect_ray = reflect(curr_dir, normal);
			hemi_sample_ray = sample_brdf(normal, curand_state, roughness);


			float _fr = fresnel(curr_dir, normal, inside_factor == 1 ? env_ior : ior, inside_factor == -1 ? ior : env_ior);
			float refl_chance = _fr;
			float refr_chance = 1.0f - _fr;
			float refl_factor = (float)reflectance / 0.5f;
			float refr_factor = (float)transparency / 0.5f;
			refl_chance = fminf(1.0f, refl_chance * refl_factor);
			refr_chance = fminf(1.0f, refr_chance * refr_factor);
			float rem = 1.0f - refl_chance - refr_chance;
			refl_chance += rem * refl_factor;
			refr_chance += rem * refr_factor;

			float ray_roll = curand_uniform(*&curand_state);
			if (refl_chance > 0.0f && ray_roll < refl_chance)
			{
				roughness = fminf(reflectance_roughness, fmaxf((1.0f - fresnel_reflectance) * reflectance_roughness, 1.0f - _fr));

				perfect_ray = reflect(curr_dir, normal);
				hemi_sample_ray = sample_brdf(normal, curand_state, roughness);
			}
			else if (refr_chance > 0.0f && ray_roll < refl_chance + refr_chance)
			{
				roughness = transparency_roughness;
				metalness = 1.0f;

				float3 n_normal = make_float3(-normal.x, -normal.y, -normal.z);
				perfect_ray = refract(curr_dir, normal, inside_factor == 1 ? env_ior : ior, inside_factor == 1 ? ior : env_ior);
				hemi_sample_ray = sample_brdf(n_normal, curand_state, roughness);
			}
			else
			{
				roughness = fminf(reflectance_roughness, fmaxf((1.0f - fresnel_reflectance) * reflectance_roughness, 1.0f - _fr));

				perfect_ray = reflect(curr_dir, normal);
				hemi_sample_ray = sample_brdf(normal, curand_state, roughness);
			}
			curr_dir = lerp(perfect_ray, hemi_sample_ray, roughness);


			cost = fmaxf(EPSILON, dot(normal, curr_dir));
			pdf = lerp(1.0f, D_PDF_CONST, roughness);
			curr_origin.x = point.x + RAY_OFFSET * curr_dir.x;
			curr_origin.y = point.y + RAY_OFFSET * curr_dir.y;
			curr_origin.z = point.z + RAY_OFFSET * curr_dir.z;

			float d_brdf_p = 1.0f / (float)M_PI;
			float d_brdf_factor = d_brdf_p * roughness;
			d_brdf.x = albedo.x * d_brdf_factor;
			d_brdf.y = albedo.y * d_brdf_factor;
			d_brdf.z = albedo.z * d_brdf_factor;

			float s_brdf_p = 1.0f / (float)cost;
			float s_brdf_factor = s_brdf_p * (1.0f - roughness);
			s_brdf.x = albedo.x * s_brdf_factor;
			s_brdf.y = albedo.y * s_brdf_factor;
			s_brdf.z = albedo.z * s_brdf_factor;

			brdf.x = d_brdf.x + s_brdf.x;
			brdf.y = d_brdf.y + s_brdf.y;
			brdf.z = d_brdf.z + s_brdf.z;



			float3 emission = albedo;
			emission.x *= emissive;
			emission.y *= emissive;
			emission.z *= emissive;


			float3 lo = estimate_direct(point, normal, scene, g_tex, curand_state, hit_obj_id, view_dir, albedo, roughness, metalness);
			color.x += throughput.x * lo.x + emission.x;
			color.y += throughput.y * lo.y + emission.y;
			color.z += throughput.z * lo.z + emission.z;

			throughput.x *= ((float)cost / (float)pdf * brdf.x + emission.x);
			throughput.y *= ((float)cost / (float)pdf * brdf.y + emission.y);
			throughput.z *= ((float)cost / (float)pdf * brdf.z + emission.z);

			if (inside_factor == -1)
			{
				throughput.x *= expf(-albedo.x * dist * absorption);
				throughput.y *= expf(-albedo.y * dist * absorption);
				throughput.z *= expf(-albedo.z * dist * absorption);
			}


			if (b >= CUTOFF_THRESHOLD)
			{
				float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
				if (curand_uniform(*&curand_state) > p)
					break;

				throughput.x /= (float)(1.0f - p);
				throughput.y /= (float)(1.0f - p);
				throughput.z /= (float)(1.0f - p);
			}
		}
		else
		{
			float3 env_color = scene->background_color;
			if (scene->env_map_status)
			{
				float3 ray = make_float3(curr_dir.x, curr_dir.y, curr_dir.z);
				env_color = tex_cube(ray, g_tex, scene);
			}
			color.x += env_color.x * throughput.x;
			color.y += env_color.y * throughput.y;
			color.z += env_color.z * throughput.z;

			break;
		}	
	}

	color.x = powf((float)color.x / (color.x + 1.0f), HDR_CONST);
	color.y = powf((float)color.y / (color.y + 1.0f), HDR_CONST);
	color.z = powf((float)color.z / (color.z + 1.0f), HDR_CONST);

	color.x = fminf(1.0f, color.x);
	color.y = fminf(1.0f, color.y);
	color.z = fminf(1.0f, color.z);

	return (color);
}


__global__ void		d_render(int *data, int2 win_wh, curandState *curand_state, gpu_scene *scene, gpu_cam *cam, gpu_tex *g_tex, int ns)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= win_wh.x || y >= win_wh.y)
		return;
	int pix = y * win_wh.x + x;

	float pix_w = 1.0f / (float)win_wh.x;
	float pix_h = 1.0f / (float)win_wh.y;

	float fx = (float)x / (float)win_wh.x * 2.0f - 1.0f;
	float fy = (float)y / (float)win_wh.y * 2.0f - 1.0f;

	float3 viewport_normal = make_float3(-cam->forward.x, -cam->forward.y, -cam->forward.z);
	float3 clr;
	clr.x = 0.0f;
	clr.y = 0.0f;
	clr.z = 0.0f;


	int i = -1;
	while (++i < ns)
	{
		float cx = fx + curand_uniform(&curand_state[pix]) * pix_w;
		float cy = fy + curand_uniform(&curand_state[pix]) * pix_h;

		float3 dir = cam->forward;
		float x_dir_factor = cam->scale * cam->aspect_ratio * cx;
		dir.x += cam->right.x * x_dir_factor;
		dir.y += cam->right.y * x_dir_factor;
		dir.z += cam->right.z * x_dir_factor;

		float y_dir_factor = cam->scale * cy;
		dir.x += cam->up.x * y_dir_factor;
		dir.y += cam->up.y * y_dir_factor;
		dir.z += cam->up.z * y_dir_factor;
		dir = normalize(dir);


		float3 frag_color = trace(cam->pos, dir, cam->z_near, cam->z_far, scene, g_tex, &curand_state[pix]);
		clr.x += frag_color.x;
		clr.y += frag_color.y;
		clr.z += frag_color.z;
	}
	clr.x = fminf(1.0f, fmaxf(0.0f, (float)clr.x / (float)ns));
	clr.y = fminf(1.0f, fmaxf(0.0f, (float)clr.y / (float)ns));
	clr.z = fminf(1.0f, fmaxf(0.0f, (float)clr.z / (float)ns));

	data[pix] = (((int)(clr.x * 255) & 0xFF) << 16) + (((int)(clr.y * 255) & 0xFF) << 8) + (((int)(clr.z * 255) & 0xFF));
}


void	PT::render()
{
	/*this->d_block_size = dim3(this->d_threads, this->d_threads);
	this->d_grid_size.x = ceilf(float(this->win_wh.x) / (float)this->d_block_size.x);
	this->d_grid_size.y = ceilf(float(this->win_wh.y) / (float)this->d_block_size.y);*/


	int ns = 32;
	d_render<<<this->d_grid_size, this->d_block_size>>>(this->d_data, this->win_wh, this->curand_state, this->d_scene, this->d_cam, this->d_tex, ns);
	if ((this->cuda_status = cudaDeviceSynchronize()) != cudaSuccess)
	std::cout << "cudaDeviceSynchronize error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';

	if ((this->cuda_status = cudaGetLastError()) != cudaSuccess)
	std::cout << "cudaDeviceSynchronize error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';

	if ((this->cuda_status = cudaMemcpy(this->h_data, this->d_data, sizeof(int) * this->win_wh.x * this->win_wh.y, cudaMemcpyDeviceToHost)) != cudaSuccess)
		std::cout << "h_data cudaMemcpy error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';
}
