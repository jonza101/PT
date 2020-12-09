
#include "pt.h"


SDL_Surface *PT::get_surf_from_pixels(void *data, int w, int h)
{
	int shift = 8;
	Uint32 r_mask = 0xFF000000 >> shift;
	Uint32 g_mask = 0x00FF0000 >> shift;
	Uint32 b_mask = 0x0000FF00 >> shift;
	Uint32 a_mask = 0x000000FF >> shift;

	int depth = 32;
	int pitch = 4 * w;

	SDL_Surface *surf = SDL_CreateRGBSurfaceFrom(data, w, h, depth, pitch, r_mask, g_mask, b_mask, a_mask);

	return (surf);
}


void		PT::merge_imgs()
{
	int img1_spp = 64;
	int img2_spp = 64;

	int img1_id = image::load_image("screenshots/pt__64.png");
	int img2_id = image::load_image("screenshots/pt__merged.png");

	const image_data *img1 = image::get_image_data(img1_id);
	const image_data *img2 = image::get_image_data(img2_id);

	int *data = (int*)malloc(sizeof(int) * img1->w * img1->h);

	int i = -1;
	while (++i < img1->w * img1->h)
	{
		int color1 = img1->data[i];
		float rc1 = (float)((color1 >> 16) & 0xFF) / 255.0f;
		float gc1 = (float)((color1 >> 8) & 0xFF) / 255.0f;
		float bc1 = (float)(color1 & 0xFF) / 255.0f;

		int color2 = img2->data[i];
		float rc2 = (float)((color2 >> 16) & 0xFF) / 255.0f;
		float gc2 = (float)((color2 >> 8) & 0xFF) / 255.0f;
		float bc2 = (float)(color2 & 0xFF) / 255.0f;

		//int rc = (int)((rc1 + rc2) * 0.5f * 255.0f);
		//int gc = (int)((gc1 + gc2) * 0.5f * 255.0f);
		//int bc = (int)((bc1 + bc2) * 0.5f * 255.0f);

		int rc = (int)((float)(rc1 * img1_spp + rc2 * img2_spp) / (float)(img1_spp + img2_spp) * 255.0f);
		int gc = (int)((float)(gc1 * img1_spp + gc2 * img2_spp) / (float)(img1_spp + img2_spp) * 255.0f);
		int bc = (int)((float)(bc1 * img1_spp + bc2 * img2_spp) / (float)(img1_spp + img2_spp) * 255.0f);

		int color = ((rc & 0xFF) << 16) + ((gc & 0xFF) << 8) + ((bc & 0xFF));
		data[i] = color;
	}

	SDL_Surface *surf = this->get_surf_from_pixels(data, img1->w, img1->h);
	IMG_SavePNG(surf, "screenshots/pt__merged.png");

	exit(0);
}


void	PT::rma_convert()
{
	int rma = image::load_image("resources/dagger_rma.png");
	const image_data *img = image::get_image_data(rma);

	int *r = (int*)malloc(sizeof(int) * img->w * img->h);
	int *m = (int*)malloc(sizeof(int) * img->w * img->h);
	int *a = (int*)malloc(sizeof(int) * img->w * img->h);

	int i = -1;
	while (++i < img->w * img->h)
	{
		int color = img->data[i];
		int rc = (color >> 16) & 0xFF;
		int gc = (color >> 8) & 0xFF;
		int bc = color & 0xFF;

		r[i] = ((rc & 0xFF) << 16) + ((rc & 0xFF) << 8) + ((rc & 0xFF));
		m[i] = ((gc & 0xFF) << 16) + ((gc & 0xFF) << 8) + ((gc & 0xFF));
		a[i] = ((bc & 0xFF) << 16) + ((bc & 0xFF) << 8) + ((bc & 0xFF));
	}


	SDL_Surface *r_surf = this->get_surf_from_pixels(r, img->w, img->h);
	SDL_Surface *m_surf = this->get_surf_from_pixels(m, img->w, img->h);
	SDL_Surface *a_surf = this->get_surf_from_pixels(a, img->w, img->h);

	IMG_SavePNG(r_surf, "resources/dagger_roughness.png");
	IMG_SavePNG(m_surf, "resources/dagger_metalness.png");
	IMG_SavePNG(a_surf, "resources/dagger_ao.png");

	exit(0);
}


void	PT::screenshot(const char *file_path)
{
	SDL_Surface *surf = this->get_surf_from_pixels(this->h_data, this->win_wh.x, this->win_wh.y);
	IMG_SavePNG(surf, file_path);
}


void	PT::update()
{
	SDL_Event sdl_event;

	Uint32 start_ticks = SDL_GetTicks();

	bool is_running = 1;
	while (is_running)
	{
		while (SDL_PollEvent(&sdl_event))
		{
			if (sdl_event.type == SDL_WINDOWEVENT)
			{
				if (sdl_event.window.event == SDL_WINDOWEVENT_CLOSE)
					is_running = 0;
			}
		}

		//this->cpy_gpu_camera();
		this->render();

		SDL_UpdateTexture(this->txt, 0, this->h_data, this->win_wh.x * sizeof(int));
		SDL_RenderCopy(this->ren, this->txt, NULL, NULL);
		SDL_RenderPresent(this->ren);


		Uint32 curr_ticks = SDL_GetTicks();
		std::cout << curr_ticks - start_ticks << " ms\n";
		start_ticks = curr_ticks;

		this->screenshot("screenshots/pt__test.png");
		exit(0);
	}
}

int		main()
{
	SDL_Init(SDL_INIT_VIDEO);
	IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG | IMG_INIT_TIF | IMG_INIT_WEBP);
	TTF_Init();

	PT *pt = new PT("PT", 1920, 1080, SDL_WINDOW_HIDDEN);//SDL_WINDOW_SHOWN);
	//pt->create_camera(make_float3(3.203f, 2.515f, 2.397f), 75.0f, make_float3(-0.7f, -0.2f, -0.7f), 0.001f, FLT_MAX);
	pt->create_camera(make_float3(0.0f, 0.0f, -3.0f), 70.0f, make_float3(0.0f, 0.0f, 1.0f), 0.001f, FLT_MAX);

	pt->init();
	pt->update();
	pt->~PT();

	return (0);
}
