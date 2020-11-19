
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


	if ((this->cuda_status = cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 16)) != cudaSuccess)
		std::cout << "cudaDeviceSetLimit error " << cudaGetErrorString(this->cuda_status) << "\n\n";

	size_t v;
	cudaDeviceGetLimit(&v, cudaLimitStackSize);
	std::cout << "stack size (Kb): " << v << '\n';


	if ((this->cuda_status = cudaMalloc((void**)&this->curand_state, sizeof(curandState) * this->win_wh.x * this->win_wh.y)) != cudaSuccess)
		std::cout << "cudaMalloc curand error " << this->cuda_status << ": " << cudaGetErrorString(this->cuda_status) << '\n';
	this->init_gpu_kernel();
}


int image::id_iter = 0;
std::vector<image_data>	image::images;
size_t image::pix = 0;


void	PT::init()
{
	this->gpu_init();


	//int rm_albedo = image::load_image("resources/rm/rm_albedo.png");
	//int rm_metalness = image::load_image("resources/rm/rm_metalness.png");
	//int rm_roughness = image::load_image("resources/rm/rm_roughness.png");
	//int rm_normal = image::load_image("resources/rm/rm_normal.png");

	int p_albedo = image::load_image("resources/p/p_albedo.png");
	int p_normal = image::load_image("resources/p/p_normal.png");

	int mt_albedo = image::load_image("resources/mt/mt_albedo.png");
	int mt_metalness = image::load_image("resources/mt/mt_metalness.png");
	int mt_roughness = image::load_image("resources/mt/mt_roughness.png");
	int mt_normal = image::load_image("resources/mt/mt_normal.png");



	//this->obj.push_back(new sphere(make_float3(0.0f, 0.0f, 0.0f),		make_float3(0.0f, -0.55f, 0.0f), 0.75f,	rm_albedo, rm_metalness, rm_roughness, rm_normal,	make_float2(1.5f, 1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.0f));
	this->obj.push_back(new sphere(make_float3(0.0f, 0.0f, 0.0f),		make_float3(0.0f, 0.0f, 0.0f), 0.75f,	p_albedo, 0.01f, 0.5f, p_normal,					make_float2(2.0f, 1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.0f));

	this->obj.push_back(new sphere(make_float3(-1.5f, -0.25f, 0.25f),	make_float3(0.0f, 0.0f, 0.0f), 0.5f,	mt_albedo, mt_metalness, mt_roughness, mt_normal,	make_float2(1.75f, 1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.0f));

	//this->obj.push_back(new sphere(make_float3(-1.0f, 1.0f, -1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.1f,	make_float3(1.0f, 1.0f, 1.0f),		0.7f, 0.2f,									make_float3(1.0f, 1.0f, 1.0f), 10.0f));
	this->obj.push_back(new sphere(make_float3(-1.0f, -0.65f, -1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.1f,	make_float3(1.0f, 1.0f, 1.0f),		0.7f, 0.2f,									make_float3(0.96f, 0.96f, 0.06f), 1.0f));

	this->obj.push_back(new sphere(make_float3(1.0f, -0.65f, -1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.1f,	make_float3(1.0f, 1.0f, 1.0f),		0.7f, 0.2f,									make_float3(0.06f, 0.54f, 0.96f), 1.0f));

	this->obj.push_back(new plane(make_float3(0.0f, -0.75f, 5.0f),		make_float3(0.0f, 1.0f, 0.0f),			make_float3(1.0f, 1.0f, 1.0f),		0.1f, 1.0f,									make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(0.0f, 9.25f, 5.0f),		make_float3(0.0f, 1.0f, 0.0f),			make_float3(1.0f, 1.0f, 0.0f),		0.1f, 1.0f,									make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(0.0f, 0.0f, -5.0f),		make_float3(0.0f, 0.0f, 1.0f),			make_float3(1.0f, 1.0f, 1.0f),		0.1f, 1.0f,									make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(0.0f, 0.0f, 5.0f),		make_float3(0.0f, 0.0f, -1.0f),			make_float3(1.0f, 1.0f, 1.0f),		0.1f, 1.0f,									make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(-3.0, 0.0f, 0.0f),		make_float3(1.0f, 0.0f, 0.0f),			make_float3(0.05f, 1.0f, 0.05f),	0.1f, 1.0f,									make_float3(0.0f, 0.0f, 0.0f)));
	this->obj.push_back(new plane(make_float3(3.0, 0.0f, 0.0f),			make_float3(-1.0f, 0.0f, 0.0f),			make_float3(1.0f, 0.05f, 0.05f),	0.1f, 1.0f,									make_float3(0.0f, 0.0f, 0.0f)));


	int i = -1;
	while (++i < this->obj.size())
	{
		if (this->obj[i]->is_light)
			this->light.push_back(this->obj[i]);
	}
	this->malloc_gpu_scene();
}

