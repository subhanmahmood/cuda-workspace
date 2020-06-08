/* Lab2Matrix.cu
 *
 *  Created on: 29 Feb 2020
 *      Author: sc01716
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#define BLOCK_SIZE 16

// Matrices are stored in row-major order
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

__global__ void MatrixMultKern(const Matrix A, const Matrix B, const Matrix C) {
	// Calculate the column index of C and B
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate the row index of C and of A
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if ((row < A.height) && (col < B.width)) {
		float Cvalue = 0;
		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < A.width; ++k) {
			Cvalue += A.elements[row * A.width + k]
					* B.elements[k * B.width + col];
		}
		C.elements[row * C.width + col] = Cvalue;
	}
}

// Matrix multiplication -Host Code
//Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatrixMult(const Matrix h_A, const Matrix h_B, Matrix h_C) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Load A and B into device memory
	Matrix d_A;
	d_A.width = h_A.width;
	d_A.height = h_A.height;
	size_t size = h_A.width * h_A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, h_A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = h_B.width;
	d_B.height = h_B.height;
	size = h_B.width * h_B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, h_B.elements, size, cudaMemcpyHostToDevice);

	// Allocate C in Device memory
	Matrix d_C;
	d_C.width = h_C.width;
	d_C.height = h_C.height;
	size = h_C.width * h_C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// Invoke Kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(d_B.width / dimBlock.x, d_A.height / dimBlock.y);
	cudaEventRecord(start);
	MatrixMultKern<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
	cudaEventRecord(stop);

	// Read C from Device to Host
	cudaMemcpy(h_C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
	//printf("Copy h_C off device: %s\n", cudaGetErrorString(err));
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time was: %f\n milliseconds", milliseconds);
	// Free Device Memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

int main(int argc, char* argv[]) {
	Matrix A, B, C;
	// Read Dimensions of A and B
	A.height = atoi(argv[1]);
	A.width = atoi(argv[2]);
	B.height = A.width;
	B.width = atoi(argv[3]);

	A.elements = (float*) malloc(A.width * A.height * sizeof(float));
	B.elements = (float*) malloc(B.width * B.height * sizeof(float));
	C.height = A.height;
	C.width = B.width;
	C.elements = (float*) malloc(C.width * C.height * sizeof(float));

	for (int i = 0; i < A.height; i++)
		for (int j = 0; j < A.width; j++)
			A.elements[i * A.width + j] = (float) (rand() % 3);
	for (int i = 0; i < B.height; i++)
		for (int j = 0; j < B.width; j++)
			B.elements[i * B.width + j] = (float) (rand() % 2);
	MatrixMult(A, B, C);
	//printing full matrix
	/*
	 for (int i = 0; i < A.height; i++) {
	 for (int j = 0; j < A.width; j++)
	 printf("%f ", A.elements[i * A.width + j]);
	 printf("\n");
	 }
	 printf("\n");
	 for (int i = 0; i < B.height; i++) {
	 for (int j = 0; j < B.width; j++)
	 printf("%f ", B.elements[i * B.width + j]);
	 printf("\n");
	 }
	 printf("\n");
	 for (int i = 0; i < C.height; i++) {
	 for (int j = 0; j < C.width; j++)
	 printf("%f ", C.elements[i * C.width + j]);
	 printf("\n");
	 }

	 printf("\n");
	 */
	//printing only part of the matrix
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++)
			printf("%f ", A.elements[i * A.width + j]);
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++)
			printf("%f ", B.elements[i * B.width + j]);
		printf("\n");
	}
	printf("\n");
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++)
			printf("%f ", C.elements[i * C.width + j]);
		printf("\n");

	}

	printf("\n");
	printf("%d ",C.width);
	return 0;
}
