#pragma once
#include "CudaParallel.cuh"
#include <iostream>

__global__ void dotProduct(float* vectorACuda, float* vectorBCuda, float* vectorCCuda, int size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size){
        vectorCCuda[i] = vectorACuda[i] * vectorBCuda[i];
    }
}


__device__ int cudaPow(int val, int pow){
    int returns = 1;
    for(int i = 0; i < pow; i++){
        returns *= val;
    }

    return returns;
}

__global__ void consolodateVector(float* vectorCuda, int level, int size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int space = cudaPow(2, level);
    int diff = cudaPow(2, level-1);
    if(i < size && (i + diff < size) && i % space == 0){
        vectorCuda[i] = vectorCuda[i] + vectorCuda[i + diff];
    }
}



CudaParallel::CudaParallel(size_t size){
    m_array_size = size;
}

float CudaParallel::dotVectors(float* vectorA, float* vectorB){
    //Create the pointers on the GPU to the data
    float* vectorACuda;
    float* vectorBCuda;
    float* vectorCCuda;

    //Create the Host Storage for the resultant vector
    float* vectorC = new float[m_array_size];

    //Malloc on the GPU for Vector A
    cudaMalloc(&vectorACuda, m_array_size * sizeof(float));

    //Malloc on the GPU for Vector B
    cudaMalloc(&vectorBCuda, m_array_size * sizeof(float));

    //Malloc on the GPU for Vector B
    cudaMalloc(&vectorCCuda, m_array_size * sizeof(float));

    //Copy vector A to the GPU
    cudaMemcpy(vectorACuda, vectorA, m_array_size * sizeof(float), cudaMemcpyHostToDevice);

    //Copy vector B to the GPU
    cudaMemcpy(vectorBCuda, vectorB, m_array_size * sizeof(float), cudaMemcpyHostToDevice);

    //Do the dot products
    dotProduct<<<std::ceil(m_array_size/256.0), 256>>>(vectorACuda, vectorBCuda, vectorCCuda, reinterpret_cast<int>(m_array_size));

    int level = 1;

    while(std::pow(2, level) <= m_array_size){
        consolodateVector<<<std::ceil(m_array_size/256.0), 256>>>(vectorCCuda, level, m_array_size);
        level++;
    }

    consolodateVector<<<std::ceil(m_array_size/256.0), 256>>>(vectorCCuda, level, m_array_size);

    cudaMemcpy(vectorC, vectorCCuda, sizeof(float), cudaMemcpyDeviceToHost);

    return *vectorC;
}

