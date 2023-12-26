#pragma once
#include "CudaParallelWrapper.hpp"
#include "CudaParallel.cuh"


CudaParallelWrapper::CudaParallelWrapper(size_t size){
    m_cudaParallel = new CudaParallel(size);
}

float* CudaParallelWrapper::dotVectors(float* vectorA, float* vectorB){
    return m_cudaParallel->dotVectors(vectorA, vectorB);
}