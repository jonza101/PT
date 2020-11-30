#pragma once

#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>

#include <cuda_runtime.h>

#include <iostream>
#include <vector>


enum tex_wrapping
{
	TW_REPEAT,
	TW_CLAMP
};


struct				gpu_tex
{
	int				count;

	int2			*wh;
	int				*data;
	size_t			*offset;
};


struct				image_data
{
	tex_wrapping	tw_mode;

	int				w;
	int				h;
	int				*data;
};

class image
{
private:
	static int						id_iter;
	static std::vector<image_data>	images;
	static size_t					pix;

	static int						env_map_status;
	static int						env_map[6];

public:
	static int						load_image(const char *file_path)
	{
		SDL_Surface *img_surf = IMG_Load(file_path);
		SDL_Surface *opt_surf = SDL_ConvertSurfaceFormat(img_surf, SDL_PIXELFORMAT_RGB888, NULL);

		if (!img_surf)
			return (-1);
		SDL_FreeSurface(img_surf);
		if (!opt_surf)
			return (-1);

		int img_id = image::id_iter;
		image_data img;
		img.w = opt_surf->w;
		img.h = opt_surf->h;
		img.data = (int*)malloc(sizeof(int) * img.w * img.h);
		SDL_memcpy(img.data, opt_surf->pixels, sizeof(int) * img.w * img.h);

		image::images.push_back(img);


		SDL_FreeSurface(opt_surf);

		image::id_iter++;
		image::pix += img.w * img.h;

		return (img_id);
	}

	static void						set_env_map(int neg_x, int pos_x, int neg_y, int pos_y, int neg_z, int pos_z)
	{
		if (neg_x < 0 || neg_x >= image::images.size() || pos_x < 0 || pos_x >= image::images.size()
				|| neg_y < 0 || neg_y >= image::images.size() || pos_y < 0 || pos_y >= image::images.size()
				|| neg_z < 0 || neg_z >= image::images.size() || pos_z < 0 || pos_z >= image::images.size())
			return;

		image::env_map_status = 1;

		image::env_map[0] = pos_x;
		image::env_map[1] = neg_x;
		image::env_map[2] = pos_y;
		image::env_map[3] = neg_y;
		image::env_map[4] = pos_z;
		image::env_map[5] = neg_z;
	}

	static int						get_env_map_status()
	{
		return (image::env_map_status);
	}

	static int						*get_env_map()
	{
		return (image::env_map);
	}


	static const image_data			*get_image_data(int img_id)
	{
		if (img_id < 0 || img_id >= image::images.size())
			return (NULL);

		return (&image::images[img_id]);
	}

	static gpu_tex					*d_malloc(gpu_tex *h_tex, cudaError_t &cuda_status)
	{
		h_tex = (gpu_tex*)malloc(sizeof(gpu_tex));

		h_tex->count = image::images.size();
		if ((cuda_status = cudaMallocManaged((void**)&h_tex->wh, sizeof(int2) * h_tex->count)) != cudaSuccess)
			std::cout << "h_tex wh cudaMallocManaged error\n";
		if ((cuda_status = cudaMallocManaged((void**)&h_tex->offset, sizeof(size_t) * h_tex->count)) != cudaSuccess)
			std::cout << "h_tex offset cudaMallocManaged error\n";

		if ((cuda_status = cudaMallocManaged((void**)&h_tex->data, sizeof(int) * image::pix)) != cudaSuccess)
			std::cout << "h_tex data cudaMallocManaged error\n";

		size_t pix_offset = 0;
		int i = -1;
		while (++i < h_tex->count)
		{
			const image_data *img = image::get_image_data(i);

			h_tex->wh[i] = make_int2(img->w, img->h);
			h_tex->offset[i] = pix_offset;
			SDL_memcpy(&h_tex->data[pix_offset], img->data, sizeof (int) * img->w * img->h);

			pix_offset += img->w * img->h;
		}

		gpu_tex *d_tex;
		if ((cuda_status = cudaMallocManaged((void**)&d_tex, sizeof(gpu_tex))) != cudaSuccess)
			std::cout << "d_tex cudaMallocManaged error\n";
		if ((cuda_status = cudaMemcpy((void**)d_tex, (const void*)h_tex, sizeof(gpu_tex), cudaMemcpyHostToDevice)) != cudaSuccess)
			std::cout << "d_tex cudaMemcpy error\n";

		return (d_tex);
	}
};
