
#include "pt.h"


void	PT::cpy_gpu_camera()
{
	this->h_cam->pos = this->cam.pos;

	this->h_cam->forward = this->cam.forward;
	this->h_cam->up = this->cam.up;
	this->h_cam->right = this->cam.right;

	this->h_cam->aspect_ratio = this->aspect_ratio;
	this->h_cam->scale = this->scale;

	this->h_cam->z_near = this->cam.z_near;
	this->h_cam->z_far = this->cam.z_far;


	if ((this->cuda_status = cudaMemcpy((void*)this->d_cam, (const void*)this->h_cam, sizeof(gpu_cam), cudaMemcpyHostToDevice)) != cudaSuccess)
		std::cout << "d_cam cudaMemcpy error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';
}


void	PT::create_lights()
{
	int i = -1;
	while (++i < this->obj.size())
	{
		if (this->obj[i]->is_light)
		{
			this->lights.push_back(new light(this->obj[i]->emission, this->obj[i]->intensity, i));
		}
	}
}

void	PT::malloc_gpu_scene()
{
	this->create_lights();


	this->h_scene = (gpu_scene*)malloc(sizeof(gpu_scene));

	int obj_count = this->obj.size();
	this->h_scene->obj_count = obj_count;
	if ((this->cuda_status = cudaMallocManaged(&this->h_scene->obj, sizeof(d_obj_data) * obj_count)) != cudaSuccess)
		std::cout << "h_scene cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';

	int light_count = this->lights.size();
	this->h_scene->light_count = light_count;
	if ((this->cuda_status = cudaMallocManaged(&this->h_scene->light, sizeof(d_light_data) * light_count)) != cudaSuccess)
		std::cout << "h_scene cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';


	this->h_scene->background_color = this->background_color;
	this->h_scene->env_map_status = image::get_env_map_status();
	if (this->h_scene->env_map_status)
	{
		if ((this->cuda_status = cudaMallocManaged(&this->h_scene->env_map, sizeof(int) * 6)) != cudaSuccess)
			std::cout << "h_scene cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';

		int *env_map = image::get_env_map();
		int i = -1;
		while (++i < 6)
			this->h_scene->env_map[i] = env_map[i];
	}


	int i = -1;
	while (++i < obj_count)
	{
		this->obj[i]->d_malloc(this->h_scene, i, this->cuda_status);
	}
	i = -1;
	while (++i < light_count)
	{
		this->lights[i]->d_malloc(this->h_scene, i, this->cuda_status);
	}


	if ((this->cuda_status = cudaMallocManaged(&this->d_scene, sizeof(gpu_scene))) != cudaSuccess)
		std::cout << "d_scene cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';
	if ((this->cuda_status = cudaMemcpy((void*)this->d_scene, (const void*)this->h_scene, sizeof(gpu_scene), cudaMemcpyHostToDevice)) != cudaSuccess)
		std::cout << "d_scene cudaMemcpy error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';


	this->h_cam = (gpu_cam*)malloc(sizeof(gpu_cam));
	if ((this->cuda_status = cudaMallocManaged(&this->d_cam, sizeof(gpu_cam))) != cudaSuccess)
		std::cout << "d_cam cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';
	this->cpy_gpu_camera();


	this->d_tex = image::d_malloc(this->h_tex, this->cuda_status);
}
