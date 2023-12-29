#pragma once

#include "ParallelLayer.cuh"
__global__ void matrix_vector_prod(float* matrixData, float* vectorData, float* outputData, int size){
    //Get the row and column
    int row = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < size){
        float output = 0;
        for(int i = 0; i < size; i++){
            output += vectorData[i] * matrixData[i + (row * size)];
        }
        outputData[row] = output;
    }
    
}

ParallelLayer::ParallelLayer(){};

void ParallelLayer::loadMatrix(float* matrixData, size_t size){
    this->size = size;

    //Malloc for the matrix
    cudaMalloc(&gpuMatrix, size * size * sizeof(float));
    
    //Memcpy the matrix to the gpu
    cudaMemcpy(gpuMatrix, matrixData, size * size * sizeof(float), cudaMemcpyHostToDevice);

    //Malloc for the vector
    cudaMalloc(&gpuVector, size * sizeof(float));
    
    //Malloc for the vector
    cudaMalloc(&gpuResultantVector, size * sizeof(float));
}

float* ParallelLayer::forwardMatrix(float* vectorData){
    //Memcpy the vector to the gpu
    cudaMemcpy(gpuVector, vectorData, size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = std::ceil(size/static_cast<float>(threads));

    dim3 THREADS{threads};
    dim3 BLOCKS{blocks};

    matrix_vector_prod<<<BLOCKS, THREADS>>>(gpuMatrix, gpuVector, gpuResultantVector, static_cast<int>(size));

    float* returns = new float[static_cast<int>(size)];

    cudaMemcpy(returns, gpuResultantVector, size * sizeof(float), cudaMemcpyDeviceToHost);

    return returns;
}