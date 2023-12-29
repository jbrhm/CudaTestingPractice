#pragma once
#include "CudaParallelWrapper.hpp"
#include "CudaParallel.cuh"


CudaParallelWrapper::CudaParallelWrapper(size_t size, float* vectorData){
    m_cudaParallel = new CudaParallel(size, vectorData);
}

float CudaParallelWrapper::dotVectors(float* vectorData){
    return m_cudaParallel->dotVectors(vectorData);
}

int CudaParallelWrapper::getCudaCores(){
    return CudaParallel::getCudaCores();
}