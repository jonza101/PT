
#include "pt.h"


void	gpu_info(int &d_threads_per_block, int &d_threads)
{
	cudaDeviceProp device_info;
	cudaGetDeviceProperties(&device_info, 0);
	d_threads_per_block = device_info.maxThreadsPerBlock;
	d_threads = sqrtf(d_threads_per_block);
	size_t max_stack_size;

	std::cout << "-------------DEVICE_INFO-----------------" << "\n\n";
	std::cout << "Name: " << device_info.name << std::endl;
	std::cout << "Core clock rate (MHz): " << device_info.clockRate * 0.001f << '\n';
	std::cout << "Mem clock rate (MHz): " << device_info.memoryClockRate * 0.001f << '\n';
	std::cout << "Bus width (bits): " << device_info.memoryBusWidth << '\n';
	std::cout << "Max threads per block: " << device_info.maxThreadsPerBlock << '\n';
	std::cout << std::endl << "-----------------------------------------" << std::endl << '\n';
}

void	PT::gpu_init()
{
	gpu_info(this->d_threads_per_block, this->d_threads);

	this->d_block_size = dim3(this->d_threads, this->d_threads);
	this->d_grid_size.x = ceilf(float(this->win_wh.x) / (float)this->d_block_size.x);
	this->d_grid_size.y = ceilf(float(this->win_wh.y) / (float)this->d_block_size.y);


	size_t v;
	cudaDeviceGetLimit(&v, cudaLimitStackSize);
	std::cout << "stack size (Kb): " << v << '\n';
	/*if ((this->cuda_status = cudaDeviceSetLimit(cudaLimitStackSize, 1 * 1024 * 128)) != cudaSuccess)
		std::cout << "cudaDeviceSetLimit error " << cudaGetErrorString(this->cuda_status) << "\n\n";*/


	if ((this->cuda_status = cudaMalloc((void**)&this->curand_state, sizeof(curandState) * this->win_wh.x * this->win_wh.y)) != cudaSuccess)
		std::cout << "cudaMalloc curand error " << this->cuda_status << ": " << cudaGetErrorString(this->cuda_status) << '\n';
	this->init_gpu_kernel();
}


void	PT::init()
{
	this->gpu_init();


	this->obj.push_back(new sphere(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), 0.75f, make_float3(1.0f, 1.0f, 1.0f), 0.15f, 0.85f, make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new sphere(make_float3(-1.5f, 0.0f, 0.25f), make_float3(0.0f, 0.0f, 0.0f), 0.5f, make_float3(0.87f, 0.16f, 0.45f), 0.25f, 0.75f, make_float3(0.0f, 0.0f, 0.0f)));

	this->obj.push_back(new sphere(make_float3(-1.0f, 1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), 0.1f, make_float3(0.87f, 0.16f, 0.45f), 0.7f, 0.2f, make_float3(1.0f, 1.0f, 1.0f)));
	this->obj[2]->set_light_propeties(1.0f, 0.5f, 1.25f);
	/*this->obj.push_back(new sphere(make_float3(1.0f, 1.0f, -1.1f), make_float3(0.0f, 0.0f, 0.0f), 0.1f, make_float3(0.87f, 0.16f, 0.45f), 0.7f, 0.2f, make_float3(0.06f, 0.54f, 0.96f)));
	this->obj[3]->set_light_propeties(1.0f, 0.5f, 1.25f);*/

	this->obj.push_back(new plane(make_float3(0.0, -0.75f, 5.0f), make_float3(0.0f, 1.0f, 0.0f), make_float3(1.0f, 1.0f, 0.05f), 0.1f, 0.9f, make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(0.0, 0.0f, 5.0f), make_float3(0.0f, 0.0f, -1.0f), make_float3(0.15f, 0.85f, 0.85f), 0.3f, 0.7f, make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(-3.0, 0.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f), make_float3(0.25f, 0.15f, 0.75f), 0.3f, 0.7f, make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(3.0, 0.0f, 0.0f), make_float3(-1.0f, 0.0f, 0.0f), make_float3(0.15f, 0.75f, 0.25f), 0.3f, 0.7f, make_float3(0.0f, 0.0f, 0.0f)));


	int i = -1;
	while (++i < this->obj.size())
	{
		if (this->obj[i]->is_light)
			this->light.push_back(this->obj[i]);
	}
	this->malloc_gpu_scene();
}

