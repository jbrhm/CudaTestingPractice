#pragma once

class CudaParallel{
private:
    int m_array_size;

public:
    CudaParallel(size_t size);

    float* dotVectors(float* vectorA, float* vectorB);
};
