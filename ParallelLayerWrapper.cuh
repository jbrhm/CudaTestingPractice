#pragma once

class ParallelLayer;

class ParallelLayerWrapper{
private:
    ParallelLayer* m_ParallelLayer;
public:
    ParallelLayerWrapper();

    ~ParallelLayerWrapper() = default;

    void loadMatrix(float* matrixData, size_t size);

    float* forwardMatrix(float* vectorData);
};