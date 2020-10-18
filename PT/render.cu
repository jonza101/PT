
#include "pt.h"


//	MATH
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__device__ float3	d_hemisphere_rand(const float3 &normal, curandState *curand_state)
{
	float3 rand_dir = make_float3(curand_uniform(*&curand_state) * 2.0f - 1.0f, curand_uniform(*&curand_state) * 2.0f - 1.0f, curand_uniform(*&curand_state));

	float3 t = cross(normal, make_float3(0.0f, 1.0f, 0.0f));
	if (length(t) == 0.0)
		t = cross(normal, make_float3(0.0f, 0.0f, 1.0f));
	t = normalize(t);
	float3 b = normalize(cross(normal, t));

	rand_dir.x = t.x * rand_dir.x + b.x * rand_dir.y + normal.x * rand_dir.z;
	rand_dir.y = t.y * rand_dir.x + b.y * rand_dir.y + normal.y * rand_dir.z;
	rand_dir.z = t.z * rand_dir.x + b.z * rand_dir.y + normal.z * rand_dir.z;
	rand_dir = normalize(rand_dir);

	return (rand_dir);
}


__device__ float3	hemisphere_rand(const float3 &normal, curandState *curand_state)
{
	float r1 = curand_uniform(*&curand_state);
	float r2 = curand_uniform(*&curand_state);

	float y = r1;
	float az = r2 * 2.0f * M_PI;
	float sin_elev = sqrtf(1.0f - y * y);
	float x = sin_elev * cosf(az);
	float z = sin_elev * sinf(az);

	float3 hemisphere_vec;
	hemisphere_vec.x = x;
	hemisphere_vec.y = y;
	hemisphere_vec.z = z;

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
	rand_dir.x = hemisphere_vec.x * b.x + hemisphere_vec.y * normal.x + hemisphere_vec.z * t.x;
	rand_dir.y = hemisphere_vec.x * b.y + hemisphere_vec.y * normal.y + hemisphere_vec.z * t.y;
	rand_dir.z = hemisphere_vec.x * b.z + hemisphere_vec.y * normal.z + hemisphere_vec.z * t.z;

	return (rand_dir);
}



__device__ int		shadow_factor(const float3 &origin, const float3 &dir, float z_near, float z_far, gpu_scene *scene, int light_id)
{
	float3 oc;

	int i = -1;
	while (++i < scene->obj_count)
	{
		if (scene->light[light_id].obj_id == i)
			continue;

		float t = -1.0f;
		switch (scene->obj[i].type)
		{
			case SPHERE:
			{
				oc.x = origin.x - scene->obj[i].pos.x;
				oc.y = origin.y - scene->obj[i].pos.y;
				oc.z = origin.z - scene->obj[i].pos.z;

				float a = dot(dir, dir);
				float b = 2.0f * dot(oc, dir);
				float c = dot(oc, oc) - scene->obj[i].radius * scene->obj[i].radius;
				float discr = b * b - 4.0f * a * c;
				float dist = (float)(-b - sqrtf(fmaxf(0.0f, discr))) / (float)(2.0f * a);
				t = -1.0f + (discr >= 0.0f) * (dist + 1.0f);
				break;
			}
			case PLANE:
			{
				float denom = dot(scene->obj[i].orientation, dir);

				oc.x = scene->obj[i].pos.x - origin.x;
				oc.y = scene->obj[i].pos.y - origin.y;
				oc.z = scene->obj[i].pos.z - origin.z;

				float tt = (float)dot(oc, scene->obj[i].orientation) / (float)(!denom ? 1.0 : denom);
				t = -1.0f + (fabs(denom) > EPSILON && tt > EPSILON) * (tt + 1.0f);
				break;
			}
		}

		if (t >= z_near && t <= z_far)
		{
			return (0);
		}
	}

	return (1);
}


__device__ int		intersection(const float3 &origin, const float3 &dir, float z_near, float z_far, gpu_scene *scene, float &dist)
{
	dist = z_far;
	int obj_id = -1;

	float3 oc;

	int i = -1;
	while (++i < scene->obj_count)
	{
		float t = -1.0f;
		switch (scene->obj[i].type)
		{
			case SPHERE:
			{
				oc.x = origin.x - scene->obj[i].pos.x;
				oc.y = origin.y - scene->obj[i].pos.y;
				oc.z = origin.z - scene->obj[i].pos.z;

				float a = dot(dir, dir);
				float b = 2.0f * dot(oc, dir);
				float c = dot(oc, oc) - scene->obj[i].radius * scene->obj[i].radius;
				float discr = b * b - 4.0f * a * c;
				float dist = (float)(-b - sqrtf(fmaxf(0.0f, discr))) / (float)(2.0f * a);
				t = -1.0f + (discr >= 0.0f) * (dist + 1.0f);
				break;
			}
			case PLANE:
			{
				float denom = dot(scene->obj[i].orientation, dir);

				oc.x = scene->obj[i].pos.x - origin.x;
				oc.y = scene->obj[i].pos.y - origin.y;
				oc.z = scene->obj[i].pos.z - origin.z;

				float tt = (float)dot(oc, scene->obj[i].orientation) / (float)(!denom ? 1.0 : denom);
				t = -1.0f + (fabs(denom) > EPSILON && tt > EPSILON) * (tt + 1.0f);
				break;
			}
		}

		if (t >= z_near && t <= z_far && t < dist)
		{
			obj_id = i;
			dist = t;
		}
	}

	return (obj_id);
}



__device__ float3	esimate_direct(const float3 &point, const float3 &normal, gpu_scene *scene, curandState *curand_state, int hit_obj_id, int light_id, float &pdf, const float3 &brdf)
{
	float3 sample_point, sample_dir;
	float3 light_dir;
	float3 li;


	int obj_id = scene->light[light_id].obj_id;
	float radius = scene->obj[obj_id].radius;

	light_dir.x = scene->obj[obj_id].pos.x - point.x;
	light_dir.y = scene->obj[obj_id].pos.y - point.y;
	light_dir.z = scene->obj[obj_id].pos.z - point.z;
	float l_dist = sqrtf(light_dir.x * light_dir.x + light_dir.y * light_dir.y + light_dir.z * light_dir.z);
	l_dist = fmaxf(l_dist, EPSILON);
	float l_dist_sqrt = sqrtf(l_dist);
	light_dir = normalize(light_dir);

	int shadow_samples = 1;
	int sh_hit = 0;
	int s = -1;
	while (++s < shadow_samples)
	{
		float r1 = curand_uniform(*&curand_state);
		float r2 = curand_uniform(*&curand_state);

		float theta = 2.0f * M_PI * r1;
		float phi = M_PI * r2;

		float x = sinf(theta) * cosf(phi) * scene->obj[obj_id].radius + scene->obj[obj_id].pos.x;
		float y = sinf(theta) * sinf(phi) * scene->obj[obj_id].radius + scene->obj[obj_id].pos.y;
		float z = cosf(theta) * scene->obj[obj_id].radius + scene->obj[obj_id].pos.z;

		sample_point.x = x;
		sample_point.y = y;
		sample_point.z = z;

		sample_dir.x = sample_point.x - point.x;
		sample_dir.y = sample_point.y - point.y;
		sample_dir.z = sample_point.z - point.z;
		sample_dir = normalize(sample_dir);

		sh_hit += shadow_factor(point, sample_dir, EPSILON * 10.0f, l_dist, scene, light_id);
	}
	float sh_factor = (float)sh_hit / (float)(shadow_samples);

	pdf = 1.0f;// / (float)(2.0f * M_PI * (1.0f - cos_phi_max));


	float l_area = 4.0f * M_PI * l_dist_sqrt * scene->obj[obj_id].radius * scene->obj[obj_id].radius;
	l_area = fmaxf(l_area, EPSILON);

	float3 intensity;
	intensity.x = scene->light[light_id].emission.x / (float)l_area;
	intensity.y = scene->light[light_id].emission.y / (float)l_area;
	intensity.z = scene->light[light_id].emission.z / (float)l_area;

	float cost = fmaxf(0.0f, dot(normal, light_dir));

	float solid_angle = (cost * l_area) / (l_dist * l_dist);
	li.x = brdf.x * scene->light[light_id].emission.x * cost * sh_factor;
	li.y = brdf.y * scene->light[light_id].emission.y * cost * sh_factor;
	li.z = brdf.z * scene->light[light_id].emission.z * cost * sh_factor;

	/*li.x = scene->obj[hit_obj_id].albedo.x * intensity.x * cost * sh_factor;
	li.y = scene->obj[hit_obj_id].albedo.y * intensity.y * cost * sh_factor;
	li.z = scene->obj[hit_obj_id].albedo.z * intensity.z * cost * sh_factor;*/

	li.x = fminf(1.0f, li.x);
	li.y = fminf(1.0f, li.y);
	li.z = fminf(1.0f, li.z);

	return (li);
}

__device__ float3	sample_light(float3 &point, float3 &normal, gpu_scene *scene, curandState *curand_state, int hit_obj_id, const float3 &brdf)
{
	float3 l;
	l.x = 0.0f;
	l.y = 0.0f;
	l.z = 0.0f;

	int i = -1;
	while (++i < scene->light_count)
	{
		int obj_id = scene->light[i].obj_id;
		if (obj_id == hit_obj_id)
			continue;

		float pdf = 0.0f;
		float3 li = esimate_direct(point, normal, scene, curand_state, hit_obj_id, i, pdf, brdf);
		if (pdf > 0.0f)
		{
			l.x += (float)li.x / (float)pdf;
			l.y += (float)li.y / (float)pdf;
			l.z += (float)li.z / (float)pdf;
		}
	}

	l.x = fminf(1.0f, fmaxf(0.0f, l.x));
	l.y = fminf(1.0f, fmaxf(0.0f, l.y));
	l.z = fminf(1.0f, fmaxf(0.0f, l.z));

	return (l);
}

__device__ float3	_trace(float3 &origin, float3 &dir, float z_near, float z_far, gpu_scene *scene, curandState *curand_state)
{
	float3 curr_origin = origin;
	float3 curr_dir = dir;

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
	float3 rand_dir;


	int bounce = -1;
	while (++bounce < 10)
	{
		float dist;
		int hit_obj_id = intersection(curr_origin, curr_dir, z_near, z_far, scene, dist);
		if (hit_obj_id >= 0)
		{
			if (scene->obj[hit_obj_id].is_light)
			{
				color.x += throughput.x * scene->obj[hit_obj_id].emission.x;
				color.y += throughput.y * scene->obj[hit_obj_id].emission.y;
				color.z += throughput.z * scene->obj[hit_obj_id].emission.z;

				break;
			}


			point.x = curr_origin.x + curr_dir.x * dist;
			point.y = curr_origin.y + curr_dir.y * dist;
			point.z = curr_origin.z + curr_dir.z * dist;

			switch (scene->obj[hit_obj_id].type)
			{
				case SPHERE:
				{
					normal.x = (float)(point.x - scene->obj[hit_obj_id].pos.x) / (float)scene->obj[hit_obj_id].radius;
					normal.y = (float)(point.y - scene->obj[hit_obj_id].pos.y) / (float)scene->obj[hit_obj_id].radius;
					normal.z = (float)(point.z - scene->obj[hit_obj_id].pos.z) / (float)scene->obj[hit_obj_id].radius;
					break;
				}
				case PLANE:
				{
					int cost = dot(curr_dir, scene->obj[hit_obj_id].orientation) > 0.0f;

					normal.x = -scene->obj[hit_obj_id].orientation.x * cost + scene->obj[hit_obj_id].orientation.x * !cost;
					normal.y = -scene->obj[hit_obj_id].orientation.y * cost + scene->obj[hit_obj_id].orientation.y * !cost;
					normal.z = -scene->obj[hit_obj_id].orientation.z * cost + scene->obj[hit_obj_id].orientation.z * !cost;
					break;
				}
			}

			float3 brdf;
			brdf.x = (float)scene->obj[hit_obj_id].albedo.x * 1.0f / (float)M_PI;
			brdf.y = (float)scene->obj[hit_obj_id].albedo.y * 1.0f / (float)M_PI;
			brdf.z = (float)scene->obj[hit_obj_id].albedo.z * 1.0f / (float)M_PI;


			rand_dir = hemisphere_rand(normal, curand_state);
			float cost = fmaxf(0.0f, dot(normal, rand_dir));
			float hemi_pdf = (float)cost / (float)M_PI;

			curr_origin = point;
			curr_dir = rand_dir;


			float3 li = sample_light(point, normal, scene, curand_state, hit_obj_id, brdf);
			color.x += throughput.x * li.x * cost;
			color.y += throughput.y * li.y * cost;
			color.z += throughput.z * li.z * cost;


			throughput.x *= (float)cost / (float)hemi_pdf * brdf.x;
			throughput.y *= (float)cost / (float)hemi_pdf * brdf.y;
			throughput.z *= (float)cost / (float)hemi_pdf * brdf.z;

			if (bounce >= 5)
			{
				float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
				if (curand_uniform(*&curand_state) > p)
					break;

				throughput.x *= 1.0f / (float)p;
				throughput.y *= 1.0f / (float)p;
				throughput.z *= 1.0f / (float)p;
			}
		}
		else
		{
			/*color.x = powf((float)color.x / (color.x + 1.0f), HDR_CONST);
			color.y = powf((float)color.y / (color.y + 1.0f), HDR_CONST);
			color.z = powf((float)color.z / (color.z + 1.0f), HDR_CONST);*/

			color.x = fminf(1.0f, color.x);
			color.y = fminf(1.0f, color.y);
			color.z = fminf(1.0f, color.z);

			return (color);
		}
	}	

	/*color.x = powf((float)color.x / (color.x + 1.0f), HDR_CONST);
	color.y = powf((float)color.y / (color.y + 1.0f), HDR_CONST);
	color.z = powf((float)color.z / (color.z + 1.0f), HDR_CONST);*/

	color.x = fminf(1.0f, color.x);
	color.y = fminf(1.0f, color.y);
	color.z = fminf(1.0f, color.z);

	return (color);
}



__device__ void		transform_ray_dir(float3 &dir, gpu_cam *cam)
{
	//	Z
	dir.x = dir.x * cosf(cam->roll) - dir.x * sinf(cam->roll);
	dir.y = dir.y * sinf(cam->roll) + dir.y * cosf(cam->roll);
	dir.z = dir.z;

	//	Y
	dir.x = dir.x * cosf(cam->yaw) + dir.z * sinf(cam->yaw);
	dir.y = dir.y;
	dir.z = dir.x * sinf(cam->yaw) + dir.z * cosf(cam->yaw);

	//	X
	dir.x = dir.x;
	dir.y = dir.y * cosf(cam->pitch) - dir.z * sinf(cam->pitch);
	dir.z = dir.y * sinf(cam->pitch) + dir.z * cosf(cam->pitch);

	dir = normalize(dir);
}

__global__ void		d_render(int *data, int2 win_wh, curandState *curand_state, gpu_scene *scene, gpu_cam *cam, int ns)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= win_wh.x || y >= win_wh.y)
		return;
	int pix = y * win_wh.x + x;



	float pix_w = 1.0f / (float)win_wh.x;
	float pix_h = 1.0f / (float)win_wh.y;

	float3 dir, r_dir;
	dir.x = (2.0f * (x + 0.5f) / (float)win_wh.x - 1) * cam->aspect_ratio * cam->scale;
	dir.y = (float)(1.0f - 2.0f * (y + 0.5f) / (float)win_wh.y) * cam->scale;
	dir.z = 1.0f;
	dir = normalize(dir);
	transform_ray_dir(dir, cam);


	float3 clr;
	clr.x = 0.0f;
	clr.y = 0.0f;
	clr.z = 0.0f;

	int i = -1;
	while (++i < ns)
	{
		/*float r_x = x - pix_w * 0.5f + (curand_uniform(&curand_state[pix]) * 2.0f - 1.0f) * pix_w;
		float r_y = y - pix_h * 0.5f + (curand_uniform(&curand_state[pix]) * 2.0f - 1.0f) * pix_h;

		r_dir.x = dir.x + (curand_uniform(&curand_state[pix]) * 2.0f - 1.0f) * pix_w;
		r_dir.y = dir.y + (curand_uniform(&curand_state[pix]) * 2.0f - 1.0f) * pix_h;
		r_dir.z = dir.z;
		r_dir = normalize(r_dir);
		transform_ray_dir(r_dir, cam);*/

		float3 frag_color = _trace(cam->pos, /*r_*/dir, cam->z_near, cam->z_far, scene, &curand_state[pix]);
		clr.x += frag_color.x;
		clr.y += frag_color.y;
		clr.z += frag_color.z;
	}
	clr.x = fminf(1.0f, fmaxf(0.0f, sqrtf((float)clr.x / (float)ns)));
	clr.y = fminf(1.0f, fmaxf(0.0f, sqrtf((float)clr.y / (float)ns)));
	clr.z = fminf(1.0f, fmaxf(0.0f, sqrtf((float)clr.z / (float)ns)));

	data[pix] = (((int)(clr.x * 255) & 0xFF) << 16) + (((int)(clr.y * 255) & 0xFF) << 8) + (((int)(clr.z * 255) & 0xFF));
}


void	PT::render()
{
	/*this->d_block_size = dim3(this->d_threads, this->d_threads);
	this->d_grid_size.x = ceilf(float(this->win_wh.x) / (float)this->d_block_size.x);
	this->d_grid_size.y = ceilf(float(this->win_wh.y) / (float)this->d_block_size.y);*/


	d_render<<<this->d_grid_size, this->d_block_size>>>(this->d_data, this->win_wh, this->curand_state, this->d_scene, this->d_cam, 32);
	if ((this->cuda_status = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << "cudaDeviceSynchronize error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';

	if ((this->cuda_status = cudaGetLastError()) != cudaSuccess)
		std::cout << "cudaDeviceSynchronize error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';

	if ((this->cuda_status = cudaMemcpy(this->h_data, this->d_data, sizeof(int) * this->win_wh.x * this->win_wh.y, cudaMemcpyDeviceToHost)) != cudaSuccess)
		std::cout << "h_data cudaMemcpy error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';

}
