
#include "pt.h"


__global__ void		init_kernel(curandState *curand_state, int2 win_wh)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x >= win_wh.x || y >= win_wh.y)
		return;
	int pix = y * win_wh.x + x;
	
	curand_init(1984 + pix, 0, 0, &curand_state[pix]);
}


void	PT::init_gpu_kernel()
{
	this->d_block_size = dim3(this->d_threads * 0.5f, this->d_threads * 0.5f);
	this->d_grid_size.x = ceilf(float(this->win_wh.x) / (float)this->d_block_size.x);
	this->d_grid_size.y = ceilf(float(this->win_wh.y) / (float)this->d_block_size.y);


	init_kernel<<<this->d_grid_size, this->d_block_size>>>(this->curand_state, this->win_wh);
	if ((this->cuda_status = cudaDeviceSynchronize()) != cudaSuccess)
		std::cout << "cudaDeviceSynchronize error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';

	if ((this->cuda_status = cudaGetLastError()) != cudaSuccess)
		std::cout << "cudaDeviceSynchronize error " << this->cuda_status << ": " << cudaGetErrorName(this->cuda_status) << ' ' << cudaGetErrorString(this->cuda_status) << '\n';
}
