#pragma once

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>


#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include <float.h>


#include <iostream>
#include <vector>


#include "matrix.h"

#include "camera.h"
#include "gpu_scene.h"
#include "image.h"

#include "sphere.h"
#include "plane.h"
#include "triangle.h"
#include "mesh.h"


#define EPSILON 0.000001f

#define PDF_CONST ((float)1.0 / (float)(2.0f * M_PI))
#define HDR_CONST (1.0f / 2.2f)



class PT
{
private:
	void	create_win(const char* title, int w, int h, int sdl_flags)
	{
		this->win = SDL_CreateWindow(title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, w, h, sdl_flags);
		this->ren = SDL_CreateRenderer(this->win, -1, SDL_RENDERER_ACCELERATED);
		this->txt = SDL_CreateTexture(this->ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);
		this->h_data = new int[w * h];
		this->win_wh.x = w;
		this->win_wh.y = h;
		this->aspect_ratio = (float)w / (float)h;

		if ((this->cuda_status = cudaMallocManaged(&this->d_data, sizeof(int) * this->win_wh.x * this->win_wh.y)) != cudaSuccess)
			std::cout << "d_data cudaMalloc error\n";

		if ((this->cuda_status = cudaMallocManaged(&this->d_data_f, sizeof(float3) * this->win_wh.x * this->win_wh.y)) != cudaSuccess)
			std::cout << "d_data_f cudaMalloc error\n";
	}

public:
	void	create_camera(const float3 &cam_pos, float cam_fov_deg, const float3 &forward, float z_near, float z_far)
	{
		this->cam = camera(cam_pos, cam_fov_deg, forward, z_near, z_far);
		this->scale = tanf((cam_fov_deg * 0.5f) * M_PI / 180.0f);
	}

	PT(const char *title, int w, int h, int sdl_flags)
	{
		this->create_win(title, w, h, sdl_flags);
	}
	~PT()
	{
		SDL_DestroyTexture(this->txt);
		SDL_DestroyRenderer(this->ren);
		SDL_DestroyWindow(this->win);
		delete this->h_data;
	}

	void					init();
	void					gpu_init();
	void					init_gpu_kernel();

	void					update();

	void					malloc_gpu_scene();
	void					cpy_gpu_camera();

	void					render();


	SDL_Surface				*get_surf_from_pixels(void *data, int w, int h);

	void					merge_imgs();
	void					rma_convert();
	void					screenshot(const char *file_path);

private:
	SDL_Window				*win;
	SDL_Renderer			*ren;
	SDL_Texture				*txt;
	int2					win_wh;

	int						*h_data;
	int						*d_data;
	float3					*d_data_f;


	camera					cam;
	float					aspect_ratio;
	float					scale;

	std::vector<shape*>		obj;
	std::vector<shape*>		light;


	cudaError_t				cuda_status;
	curandState				*curand_state;

	int						d_threads_per_block;
	int						d_threads;
	dim3					d_block_size;
	dim3					d_grid_size;

	gpu_scene				*h_scene;
	gpu_scene				*d_scene;

	gpu_cam					*h_cam;
	gpu_cam					*d_cam;

	gpu_tex					*h_tex;
	gpu_tex					*d_tex;
};


