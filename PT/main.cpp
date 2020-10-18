
#include "pt.h"


void	PT::update()
{
	SDL_Event sdl_event;

	Uint32 start = SDL_GetTicks();

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

		this->cpy_gpu_camera();
		this->render();

		SDL_UpdateTexture(this->txt, 0, this->h_data, this->win_wh.x * sizeof(int));
		SDL_RenderCopy(this->ren, this->txt, NULL, NULL);
		SDL_RenderPresent(this->ren);

		Uint32 temp = SDL_GetTicks();
		std::cout << temp - start << " ms\n";
		start = temp;
	}
}

int		main()
{
	SDL_Init(SDL_INIT_VIDEO);
	IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG | IMG_INIT_TIF | IMG_INIT_WEBP);
	TTF_Init();

	PT *pt = new PT("PT", 1280, 720, SDL_WINDOW_SHOWN);
	pt->create_camera(make_float3(0.0f, 0.0f, -3.0f), 70.0f, make_float3(0.0f, 0.0f, 1.0f), 0.01f, 1000.0f);

	pt->init();
	pt->update();
	pt->~PT();

	return (0);
}
