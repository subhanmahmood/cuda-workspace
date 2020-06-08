#include <iostream>
#include <math.h>
// function to add the elements of two arrays
__global__ void add(int n, float *x, float *y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	y[index] = x[index] + y[index];
}
int main(void) {

	int N = 1 << 20; // 1M elements
	float *x, *y;
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));
// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
// Run kernel on 1M elements on the CPU
	clock_t start = clock();
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add<<<numBlocks, blockSize>>>(N, x, y);

	//wait for GPU to finish before accessing host
	cudaDeviceSynchronize();
	clock_t stop = clock();
	double elapsed = (double)(stop - start) * 1000.0 /
	CLOCKS_PER_SEC;
	printf("Time elapsed in ms: %f", elapsed);
	printf("HOE!!!!");
// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	std::cout << "\nMax error: " << maxError << std::endl;
// Free memory
	cudaFree(x);
	cudaFree(y);
	return 0;
}
