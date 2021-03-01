
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
	std::cout << "Saved to " << file_path << '\n';

	exit(0);
}


void	PT::update()
{
	Uint32 start_ticks = SDL_GetTicks();

	
	//this->cpy_gpu_camera();
	this->render();

	SDL_UpdateTexture(this->txt, 0, this->h_data, this->win_wh.x * sizeof(int));
	SDL_RenderCopy(this->ren, this->txt, NULL, NULL);
	SDL_RenderPresent(this->ren);

	Uint32 curr_ticks = SDL_GetTicks();
	std::cout << curr_ticks - start_ticks << " ms\n";
	start_ticks = curr_ticks;


	this->screenshot(this->screenshot_path);
}

int		main()
{
	SDL_Init(SDL_INIT_VIDEO);
	IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG | IMG_INIT_TIF | IMG_INIT_WEBP);
	TTF_Init();

	PT *pt = new PT("PT", 1920, 1080, SDL_WINDOW_HIDDEN);//SDL_WINDOW_SHOWN);

	pt->init();
	pt->update();
	pt->~PT();

	return (0);
}
