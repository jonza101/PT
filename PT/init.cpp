
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
	//int icosphere_id = mesh::load_mesh("resources/mesh/icosphere.obj");
	//int spellbook_id = mesh::load_mesh("resources/mesh/spellbook.obj");
	int quad_id = mesh::load_mesh("resources/mesh/quad.obj");


	//int claymore_id = mesh::load_mesh("resources/claymore/claymore.obj");
	//int floor_id = mesh::load_mesh("resources/floor/floor.obj");

	//int greathelm_id = mesh::load_mesh("resources/greathelm/greathelm.obj");
	//int book_id = mesh::load_mesh("resources/book/book.obj");
	//int goblet_id = mesh::load_mesh("resources/goblet/goblet.obj");
	//int pitcher_id = mesh::load_mesh("resources/pitcher/pitcher.obj");
	//int dagger_id = mesh::load_mesh("resources/dagger/dagger.obj");
	//int stack_id = mesh::load_mesh("resources/coins/stack.obj");
	//int table_id = mesh::load_mesh("resources/table/table.obj");

	//int tapestry_id = mesh::load_mesh("resources/tapestry/tapestry.obj");
	//int torch_tall_id = mesh::load_mesh("resources/torch/torch_tall.obj");
	//int wall_id = mesh::load_mesh("resources/wall/wall.obj");
	//int wall_support_id = mesh::load_mesh("resources/wall_support/wall_support.obj");


	//int nx = image::load_image("resources/cubemap/lost_city/nx.png");
	//int ny = image::load_image("resources/cubemap/lost_city/ny.png");
	//int nz = image::load_image("resources/cubemap/lost_city/nz.png");
	//int px = image::load_image("resources/cubemap/lost_city/px.png");
	//int py = image::load_image("resources/cubemap/lost_city/py.png");
	//int pz = image::load_image("resources/cubemap/lost_city/pz.png");
	//image::set_env_map(nx, px, ny, py, nz, pz);

	this->background_color = make_float3(0.64f, 0.67f, 0.68f);


	int p_albedo = image::load_image("resources/p/p_albedo.png");
	int p_normal = image::load_image("resources/p/p_normal.png");
	
	int mt_albedo = image::load_image("resources/mt/mt_albedo.png");
	int mt_metalness = image::load_image("resources/mt/mt_metalness.png");
	int mt_roughness = image::load_image("resources/mt/mt_roughness.png");
	int mt_normal = image::load_image("resources/mt/mt_normal.png");


	//int claymore_albedo = image::load_image("resources/claymore/claymore_albedo.png");
	//int claymore_metalness = image::load_image("resources/claymore/claymore_metalness.png");
	//int claymore_roughness = image::load_image("resources/claymore/claymore_roughness.png");
	//int claymore_normal = image::load_image("resources/claymore/claymore_normal.png");
	//
	//int floor_albedo = image::load_image("resources/floor/floor_albedo.png");
	//int floor_metalness = image::load_image("resources/floor/floor_metalness.png");
	//int floor_roughness = image::load_image("resources/floor/floor_roughness.png");
	//int floor_normal = image::load_image("resources/floor/floor_normal.png");
	//
	//int greathelm_albedo = image::load_image("resources/greathelm/greathelm_albedo.png");
	//int greathelm_metalness = image::load_image("resources/greathelm/greathelm_metalness.png");
	//int greathelm_roughness = image::load_image("resources/greathelm/greathelm_roughness.png");
	//int greathelm_normal = image::load_image("resources/greathelm/greathelm_normal.png");
	//
	//int book_albedo = image::load_image("resources/book/book_albedo.png");
	//int book_roughness = image::load_image("resources/book/book_roughness.png");
	//int book_normal = image::load_image("resources/book/book_normal.png");
	//
	//int goblet_albedo = image::load_image("resources/goblet/goblet_albedo.png");
	//int goblet_metalness = image::load_image("resources/goblet/goblet_metalness.png");
	//int goblet_roughness = image::load_image("resources/goblet/goblet_roughness.png");
	//int goblet_normal = image::load_image("resources/goblet/goblet_normal.png");
	//
	//int pitcher_albedo = image::load_image("resources/pitcher/pitcher_albedo.png");
	//int pitcher_metalness = image::load_image("resources/pitcher/pitcher_metalness.png");
	//int pitcher_roughness = image::load_image("resources/pitcher/pitcher_roughness.png");
	//int pitcher_normal = image::load_image("resources/pitcher/pitcher_normal.png");
	//
	//int dagger_albedo = image::load_image("resources/dagger/dagger_albedo.png");
	//int dagger_metalness = image::load_image("resources/dagger/dagger_metalness.png");
	//int dagger_roughness = image::load_image("resources/dagger/dagger_roughness.png");
	//int dagger_normal = image::load_image("resources/dagger/dagger_normal.png");
	//
	//int coins_albedo = image::load_image("resources/coins/coins_albedo.png");
	//int coins_roughness = image::load_image("resources/coins/coins_roughness.png");
	//int coins_normal = image::load_image("resources/coins/coins_normal.png");
	//
	//int table_albedo = image::load_image("resources/table/table_albedo.png");
	//int table_roughness = image::load_image("resources/table/table_roughness.png");
	//int table_normal = image::load_image("resources/table/table_normal.png");
	//
	//int tapestry_albedo = image::load_image("resources/tapestry/tapestry_albedo1.png");
	//int tapestry_normal = image::load_image("resources/tapestry/tapestry_normal.png");
	//
	//int torch_albedo = image::load_image("resources/torch/torch_albedo.png");
	//int torch_metalness = image::load_image("resources/torch/torch_metalness.png");
	//int torch_roughness = image::load_image("resources/torch/torch_roughness.png");
	//int torch_normal = image::load_image("resources/torch/torch_normal.png");
	//
	//int wall_albedo = image::load_image("resources/wall/wall_albedo.png");
	//int wall_roughness = image::load_image("resources/wall/wall_roughness.png");
	//int wall_normal = image::load_image("resources/wall/wall_normal.png");



	this->obj.push_back(new sphere(make_float3(0.0f, 0.0f, 0.0f),		make_float3(0.0f, 0.0f, 0.0f), 0.75f,	p_albedo, 0.01f, 0.5f, p_normal,					make_float2(2.0f, 1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.0f));
	this->obj.push_back(new sphere(make_float3(-1.5f, -0.25f, 0.25f),	make_float3(0.0f, 0.0f, 0.0f), 0.5f,	mt_albedo, mt_metalness, mt_roughness, mt_normal,	make_float2(1.75f, 1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.0f));

	//this->obj.push_back(new sphere(make_float3(-1.0f, 1.0f, -1.0f),		make_float3(0.0f, 0.0f, 0.0f), 0.1f,	make_float3(1.0f, 1.0f, 1.0f),		0.7f, 0.2f,									make_float3(1.0f, 1.0f, 1.0f), 10.0f));
	//this->obj.push_back(new sphere(make_float3(-1.0f, -0.65f, -1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.1f,	make_float3(1.0f, 1.0f, 1.0f),		0.7f, 0.2f,									make_float3(0.96f, 0.96f, 0.06f), 1.0f));

	//this->obj.push_back(new sphere(make_float3(1.0f, -0.65f, -1.0f),	make_float3(0.0f, 0.0f, 0.0f), 0.1f,	make_float3(1.0f, 1.0f, 1.0f),		0.7f, 0.2f,									make_float3(0.06f, 0.54f, 0.96f), 1.0f));

	this->lights.push_back(new light(make_float3(1.0f, 1.0f, 1.0f), 1.0f, make_float3(1.0f, -0.35f, 0.5f), 0.125f));


	mesh::create_mesh(quad_id, make_float3(0.0f, -0.75f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 1.0f, 10.0f), make_float3(1.0f, 1.0f, 1.0f), 0.01f, 1.0f, this->obj);

	//this->obj.push_back(new plane(make_float3(0.0f, -0.75f, 5.0f),		make_float3(0.0f, 1.0f, 0.0f),			make_float3(1.0f, 1.0f, 1.0f),		0.1f, 1.0f));
	//this->obj.push_back(new plane(make_float3(0.0f, 9.25f, 5.0f),		make_float3(0.0f, 1.0f, 0.0f),			make_float3(1.0f, 1.0f, 0.0f),		0.1f, 1.0f));
	//this->obj.push_back(new plane(make_float3(0.0f, 0.0f, -5.0f),		make_float3(0.0f, 0.0f, 1.0f),			make_float3(1.0f, 1.0f, 1.0f),		0.1f, 1.0f));
	//this->obj.push_back(new plane(make_float3(0.0f, 0.0f, 5.0f),		make_float3(0.0f, 0.0f, -1.0f),			make_float3(1.0f, 1.0f, 1.0f),		0.1f, 1.0f));
	//this->obj.push_back(new plane(make_float3(-3.0, 0.0f, 0.0f),		make_float3(1.0f, 0.0f, 0.0f),			make_float3(0.05f, 1.0f, 0.05f),	0.1f, 1.0f));
	//this->obj.push_back(new plane(make_float3(3.0, 0.0f, 0.0f),			make_float3(-1.0f, 0.0f, 0.0f),			make_float3(1.0f, 0.05f, 0.05f),	0.1f, 1.0f));

	//mesh::create_mesh(icosphere_id, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.75f, 0.75f, 0.75f), make_float3(1.0f, 0.05f, 1.0f), 1.0f, 0.15f, this->obj);
	//mesh::create_mesh(spellbook_id, make_float3(0.0f, -0.55f, 0.0f), make_float3(0.0f, 1.3f, 0.0f), make_float3(0.2f, 0.2f, -0.2f), s_albedo, s_metalness, s_roughness, s_normal, this->obj);
	//mesh::create_mesh(quad_id, make_float3(0.0f, -0.75f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(3.0f, 2.0f, 2.0f), make_float3(1.0f, 1.0f, 1.0f), 0.01f, 1.0f, this->obj);


	//this->obj.push_back(new sphere(make_float3(-4.7f, 6.2f, 2.6f), make_float3(0.0f, 0.0f, 0.0f), 0.5f, make_float3(1.0f, 1.0f, 1.0f), 1.0f, 1.0f, make_float3(1.0f, 0.25f, 0.0f), 128.0f));
	
	//mesh::create_mesh(wall_id, make_float3(0.0f, -0.75f, -0.95f), make_float3(0.0f, 3.14f, 0.0f), make_float3(0.12f, 0.075f, 0.12f), wall_albedo, 0.001f, wall_roughness, wall_normal, this->obj);
	//mesh::create_mesh(wall_id, make_float3(-5.0f, -0.75f, 2.33f), make_float3(0.0f, 4.712f, 0.0f), make_float3(0.1f, 0.075f, 0.2f), wall_albedo, 0.001f, wall_roughness, wall_normal, this->obj);
	//mesh::create_mesh(floor_id, make_float3(5.58f, 0.43f, 0.0f), make_float3(0.0f, 0.0f, 1.57f), make_float3(10.0f, 1.0f, 10.0f), wall_albedo, 0.001f, wall_roughness, wall_normal, this->obj);
	//mesh::create_mesh(floor_id, make_float3(-5.95f, 0.43f, 0.0f), make_float3(0.0f, 0.0f, 4.712f), make_float3(10.0f, 1.0f, 10.0f), wall_albedo, 0.001f, wall_roughness, wall_normal, this->obj);
	//mesh::create_mesh(wall_support_id, make_float3(-4.7f, -0.75f, -0.95f), make_float3(0.0f, 3.14f, 0.0f), make_float3(0.1f, 0.12f, 0.1f), wall_albedo, 0.001f, wall_roughness, wall_normal, this->obj);
	//mesh::create_mesh(wall_support_id, make_float3(4.7f, -0.75f, -0.95f), make_float3(0.0f, 3.14f, 0.0f), make_float3(0.1f, 0.12f, 0.1f), wall_albedo, 0.001f, wall_roughness, wall_normal, this->obj);
	//mesh::create_mesh(tapestry_id, make_float3(0.0f, 2.35f, -1.259f), make_float3(0.0f, 3.14f, 0.0f), make_float3(0.024f, 0.024f, 0.024f), tapestry_albedo, 0.001f, 1.0f, tapestry_normal, this->obj);
	
	//mesh::create_mesh(floor_id, make_float3(0.0f, -0.75f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(10.0f, 1.0f, 10.0f), floor_albedo, floor_metalness, floor_roughness, floor_normal, this->obj);
	//mesh::create_mesh(floor_id, make_float3(0.0f, 10.4f, 0.0f), make_float3(0.0f, 0.0f, 3.14f), make_float3(10.0f, 1.0f, 10.0f), wall_albedo, 0.001f, wall_roughness, wall_normal, this->obj);
	
	//mesh::create_mesh(table_id, make_float3(0.0f, 0.465f, 0.12f), make_float3(0.0f, 3.14f, 0.0f), make_float3(1000.0f, 1173.0f, 1150.0f), table_albedo, 0.001f, table_roughness, table_normal, this->obj);
	
	//mesh::create_mesh(claymore_id, make_float3(3.05f, 2.803f, -0.688f), make_float3(2.967f, 0.0f, 0.0f), make_float3(0.03f, 0.03f, 0.03f), claymore_albedo, claymore_metalness, claymore_roughness, claymore_normal, this->obj);
	//mesh::create_mesh(torch_tall_id, make_float3(-4.7f, -0.81f, 2.6f), make_float3(0.0f, 0.785f, 0.0f), make_float3(0.065f, 0.07f, 0.065f), torch_albedo, torch_metalness, torch_roughness, torch_normal, this->obj);
	
	//mesh::create_mesh(greathelm_id, make_float3(1.913f, 1.532f, 0.7f), make_float3(0.349f, 3.4f, 0.0f), make_float3(0.03f, 0.03f, 0.03f), greathelm_albedo, greathelm_metalness, greathelm_roughness, greathelm_normal, this->obj);
	//mesh::create_mesh(book_id, make_float3(-0.009f, 1.785f, 0.677f), make_float3(0.0f, 0.0f, 0.0f), make_float3(75.0f, 75.0f, 75.0f), book_albedo, 0.001f, book_roughness, book_normal, this->obj);
	//mesh::create_mesh(goblet_id, make_float3(-1.184f, 1.678f, 0.386f), make_float3(0.0f, 0.0f, 0.0f), make_float3(0.4f, 0.4f, 0.4f), goblet_albedo, goblet_metalness, goblet_roughness, goblet_normal, this->obj);
	//mesh::create_mesh(pitcher_id, make_float3(-1.441f, 2.331f, -0.303f), make_float3(0.0f, 4.188f, 0.0f), make_float3(0.45f, 0.45f, 0.45f), pitcher_albedo, pitcher_metalness, pitcher_roughness, pitcher_normal, this->obj);
	//mesh::create_mesh(dagger_id, make_float3(1.196f, 1.711f, 0.75f), make_float3(0.0f, 2.879f, 0.0f), make_float3(0.022f, 0.022f, 0.022f), dagger_albedo, dagger_metalness, dagger_roughness, dagger_normal, this->obj);
	//mesh::create_mesh(stack_id, make_float3(-0.75f, 1.6815f, -0.52f), make_float3(0.0f, 0.0f, 0.0f), make_float3(3.0f, 3.0f, 3.0f), coins_albedo, 1.0f, coins_roughness, coins_normal, this->obj);



	//this->rma_convert();
	//this->merge_imgs();


	this->malloc_gpu_scene();


	size_t free, total;
	cudaMemGetInfo(&free, &total);
	std::cout << free << " / " << total << " (bytes)\n";
}
