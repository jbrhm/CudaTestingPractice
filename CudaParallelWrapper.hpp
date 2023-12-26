#pragma once
#include <assert.h>
#include <stddef.h>

class CudaParallel;

class CudaParallelWrapper{
private:
    CudaParallel* m_cudaParallel;
public:
    CudaParallelWrapper(size_t size);

    ~CudaParallelWrapper() = default;

    float* dotVectors(float* vectorA, float* vectorB);
};