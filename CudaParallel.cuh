#pragma once

class CudaParallel{
private:
    int m_array_size;

    //Cuda Vector Ptrs
    float* vectorConstantCuda;
    float* vectorVariableCuda;
    float* vectorOutputCuda;

    //Host Vector Output
    float* vectorOutput;

public:
    CudaParallel(size_t size, float* vectorData);

    static int getCudaCores();

    float dotVectors(float* vectorData);
};
