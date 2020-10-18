
#include "pt.h"


void	PT::cpy_gpu_camera()
{
	this->h_cam->pos = this->cam.pos;
	
	this->h_cam->pitch = this->cam.pitch;
	this->h_cam->yaw = this->cam.yaw;
	this->h_cam->roll = this->cam.roll;

	this->h_cam->aspect_ratio = this->aspect_ratio;
	this->h_cam->scale = this->scale;

	this->h_cam->z_near = this->cam.z_near;
	this->h_cam->z_far = this->cam.z_far;


	if ((this->cuda_status = cudaMemcpy((void*)this->d_cam, (const void*)this->h_cam, sizeof(gpu_cam), cudaMemcpyHostToDevice)) != cudaSuccess)
		std::cout << "d_cam cudaMemcpy error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';
}

void	PT::malloc_gpu_scene()
{
	this->h_scene = (gpu_scene*)malloc(sizeof(gpu_scene));

	int obj_count = this->obj.size();
	this->h_scene->obj_count = obj_count;
	if ((this->cuda_status = cudaMallocManaged(&this->h_scene->obj, sizeof(d_obj_data) * obj_count)) != cudaSuccess)
		std::cout << "h_scene cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';

	int light_count = this->light.size();
	this->h_scene->light_count = light_count;
	if ((this->cuda_status = cudaMallocManaged(&this->h_scene->light, sizeof(d_light_data) * light_count)) != cudaSuccess)
		std::cout << "h_scene cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';


	int l = 0;
	int i = -1;
	while (++i < obj_count)
	{
		this->obj[i]->d_malloc(this->h_scene, i, this->cuda_status);
		if (this->obj[i]->is_light)
		{
			this->obj[i]->d_malloc_light(this->h_scene, i, l, this->cuda_status);
			l++;
		}
	}


	if ((this->cuda_status = cudaMallocManaged(&this->d_scene, sizeof(gpu_scene))) != cudaSuccess)
		std::cout << "d_scene cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';
	if ((this->cuda_status = cudaMemcpy((void*)this->d_scene, (const void*)this->h_scene, sizeof(gpu_scene), cudaMemcpyHostToDevice)) != cudaSuccess)
		std::cout << "d_scene cudaMemcpy error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';


	this->h_cam = (gpu_cam*)malloc(sizeof(gpu_cam));
	if ((this->cuda_status = cudaMallocManaged(&this->d_cam, sizeof(gpu_cam))) != cudaSuccess)
		std::cout << "d_cam cudaMallocManaged error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << '\n';
}
