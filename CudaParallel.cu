#pragma once
#include "CudaParallel.cuh"

__global__ void dotProduct(float* vectorACuda, float* vectorBCuda, float* vectorCCuda, int size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size){
        vectorCCuda[i] = vectorACuda[i] * vectorBCuda[i];
    }
}

// __global__ void CudaParallel::consolodateVector(float* vectorCuda, int level){
//     if()

//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int space = cudaPow(2, level);
//     if(i < m_array_size){
        
//     }
// }

__device__ int cudaPow(int val, int pow){
    int returns = 1;
    for(int i = 0; i < pow; i++){
        returns *= val;
    }

    return returns;
}


CudaParallel::CudaParallel(size_t size){
    m_array_size = size;
}

float* CudaParallel::dotVectors(float* vectorA, float* vectorB){
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

    cudaMemcpy(vectorC, vectorCCuda, m_array_size * sizeof(float), cudaMemcpyDeviceToHost);

    return vectorC;
}

