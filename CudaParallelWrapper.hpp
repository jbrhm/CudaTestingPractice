#pragma once
#include <assert.h>
#include <stddef.h>

class CudaParallel;

class CudaParallelWrapper{
private:
    CudaParallel* m_cudaParallel;
public:
    CudaParallelWrapper(size_t size, float* vectorData);

    ~CudaParallelWrapper() = default;

    float dotVectors(float* vectorData);

    int getCudaCores();
};