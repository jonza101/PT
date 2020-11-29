
#include "pt.h"


void	PT::screenshot(const char *file_path)
{
	SDL_Texture *target = SDL_GetRenderTarget(this->ren);
	SDL_SetRenderTarget(this->ren, this->txt);
	int w, h;
	SDL_QueryTexture(this->txt, NULL, NULL, &w, &h);
	SDL_Surface *sur = SDL_CreateRGBSurface(0, w, h, 32, 0, 0, 0, 0);
	SDL_RenderReadPixels(this->ren, NULL, sur->format->format, sur->pixels, sur->pitch);
	IMG_SavePNG(sur, file_path);
	SDL_FreeSurface(sur);
	SDL_SetRenderTarget(this->ren, target);
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

		/*this->screenshot("screenshots/pt_.png");
		exit(0);*/
	}
}

int		main()
{
	SDL_Init(SDL_INIT_VIDEO);
	IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG | IMG_INIT_TIF | IMG_INIT_WEBP);
	TTF_Init();

	PT *pt = new PT("PT", 1280, 720, SDL_WINDOW_SHOWN);
	pt->create_camera(make_float3(0.0f, 0.5f, -3.0f), 70.0f, make_float3(0.0f, -0.1f, 1.0f), 0.001f, FLT_MAX);

	pt->init();
	pt->update();
	pt->~PT();

	return (0);
}
