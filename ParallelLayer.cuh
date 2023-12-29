#pragma once
#include <stddef.h>

class ParallelLayer{
private:

    //GPU Ptrs
    float* gpuMatrix;
    float* gpuVector;
    float* gpuResultantVector;

    size_t size;

public:
    ParallelLayer();

    ~ParallelLayer() = default;

    void loadMatrix(float* matrixData, size_t size);

    float* forwardMatrix(float* vectorData);
};