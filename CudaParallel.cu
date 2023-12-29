#pragma once
#include "CudaParallel.cuh"
#include <iostream>
#include "helper_cuda.h"

__global__ void dotProduct(float* vectorACuda, float* vectorBCuda, float* vectorCCuda, int size){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < size){
        vectorCCuda[i] = vectorACuda[i] * vectorBCuda[i];
        i += blockDim.x * 256;//This is thread/block
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



CudaParallel::CudaParallel(size_t size, float* vectorData){
    m_array_size = size;

    //Malloc on the GPU for Vector A
    cudaMalloc(&vectorConstantCuda, m_array_size * sizeof(float));
    //Memcpy the data to the GPU
    cudaMemcpy(vectorConstantCuda, vectorData, m_array_size * sizeof(float), cudaMemcpyHostToDevice);

    //Malloc on the GPU for Vector B
    cudaMalloc(&vectorVariableCuda, m_array_size * sizeof(float));

    //Malloc on the GPU for Vector B
    cudaMalloc(&vectorOutputCuda, m_array_size * sizeof(float));

    //Create the Host Storage for the resultant vector
    vectorOutput = new float[m_array_size];
}

// you must first call the cudaGetDeviceProperties() function, then pass 
// the devProp structure returned to this function:
int CudaParallel::getCudaCores(){  
    int deviceID;
    cudaDeviceProp props;

    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);
        
    int CUDACores = _ConvertSMVer2Cores(props.major, props.minor) * props.multiProcessorCount;

    return CUDACores;
}

float CudaParallel::dotVectors(float* vectorData){

    //Memcpy the output vector
    cudaMemcpy(vectorVariableCuda, vectorData, m_array_size * sizeof(float), cudaMemcpyHostToDevice);

    //Do the dot products
    dotProduct<<<std::ceil(1024/256.0), 256>>>(vectorConstantCuda, vectorVariableCuda, vectorOutputCuda, reinterpret_cast<int>(m_array_size));

    int level = 1;

    while(std::pow(2, level) <= m_array_size){
        consolodateVector<<<std::ceil(m_array_size/256.0), 256>>>(vectorOutputCuda, level, m_array_size);
        level++;
    }

    consolodateVector<<<std::ceil(m_array_size/256.0), 256>>>(vectorOutputCuda, level, m_array_size);

    float* resultant = new float();

    cudaMemcpy(resultant, vectorOutputCuda, sizeof(float), cudaMemcpyDeviceToHost);

    return *resultant;
}

