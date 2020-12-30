
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


	if ((this->cuda_status = cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 32)) != cudaSuccess)
		std::cout << "cudaDeviceSetLimit error " << cudaGetErrorString(this->cuda_status) << "\n\n";

	size_t v;
	cudaDeviceGetLimit(&v, cudaLimitStackSize);
	std::cout << "stack size (Kb): " << v << '\n';


	if ((this->cuda_status = cudaMalloc((void**)&this->curand_state, sizeof(curandState) * this->win_wh.x * this->win_wh.y)) != cudaSuccess)
		std::cout << "cudaMalloc curand error " << this->cuda_status << ": " << cudaGetErrorString(this->cuda_status) << '\n';
	this->init_gpu_kernel();
}


int							image::id_iter = 0;
std::vector<image_data>		image::images;
size_t						image::pix = 0;
int							image::env_map_status = 0;
int							image::env_map[6];

int							mesh::id_iter = 0;
std::vector<mesh_data>		mesh::meshes;



void	PT::init()
{
	this->gpu_init();


	//int cube_id = mesh::load_mesh("resources/mesh/cube.obj");
	int quad_id = mesh::load_mesh("resources/mesh/quad.obj");


	//int nx = image::load_image("resources/cubemap/lost_city/nx.png");
	//int ny = image::load_image("resources/cubemap/lost_city/ny.png");
	//int nz = image::load_image("resources/cubemap/lost_city/nz.png");
	//int px = image::load_image("resources/cubemap/lost_city/px.png");
	//int py = image::load_image("resources/cubemap/lost_city/py.png");
	//int pz = image::load_image("resources/cubemap/lost_city/pz.png");
	//image::set_env_map(nx, px, ny, py, nz, pz);

	this->background_color = make_float3(0.64f, 0.67f, 0.68f);
	this->lights.push_back(new light(make_float3(1.0f, 1.0f, 1.0f), 1.0f, make_float3(1.0f, -0.35f, 0.5f), 0.125f));


	int p_albedo = image::load_image("resources/p/p_albedo.png");
	int p_normal = image::load_image("resources/p/p_normal.png");
	
	int mt_albedo = image::load_image("resources/mt/mt_albedo.png");
	int mt_metalness = image::load_image("resources/mt/mt_metalness.png");
	int mt_roughness = image::load_image("resources/mt/mt_roughness.png");
	int mt_normal = image::load_image("resources/mt/mt_normal.png");



	this->obj.push_back(new sphere(make_float3(0.0f, 0.0f, 0.0f),		make_float3(0.0f, 0.0f, 0.0f), 0.75f,	p_albedo, 0.01f, 0.5f, p_normal,					make_float2(2.0f, 1.0f)));
	this->obj.push_back(new sphere(make_float3(-1.5f, -0.25f, 0.25f),	make_float3(0.0f, 0.0f, 0.0f), 0.5f,	mt_albedo, mt_metalness, mt_roughness, mt_normal,	make_float2(1.75f, 1.0f)));

	//this->obj.push_back(new sphere(make_float3(-1.0f, 1.0f, -1.0f),		make_float3(0.0f, 0.0f, 0.0f), 0.1f,	mt_albedo, mt_metalness, mt_roughness, mt_normal,	make_float2(4.0f, 2.0f),			1, 10.0f));
	
	//this->obj.push_back(new sphere(make_float3(1.0f, -0.65f, -1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.1f,	cg_albedo, 1.0f, cg_roughness, cg_normal,			make_float2(2.0f, 1.0f),			1, 2.0f));
	//this->obj.push_back(new sphere(make_float3(-1.0f, -0.65f, -1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.1f,	cv_albedo, 1.0f, cv_roughness, cv_normal,			make_float2(2.0f, 1.0f),			1, 2.0f));

	mesh::create_mesh(quad_id, make_float3(0.0f, -0.75f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 1.0f, 10.0f), make_float3(1.0f, 1.0f, 1.0f), 0.01f, 1.0f, this->obj);




	//this->rma_convert();
	//this->merge_imgs();


	this->malloc_gpu_scene();


	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << free << " / " << total << " (bytes)\n";
}
