#include "CudaParallelWrapper.hpp"
#include "ParallelLayerWrapper.cuh"
#include <cstddef>
#include <iostream>

int main(){
    size_t size = 1000;

    float* inputVector = new float[size];
    for(int i = 0; i < size; i++){
        inputVector[i] = 2;
    }

    float* matrixData = new float[size * size];
    for(int i = 0; i < size * size; i++){
        matrixData[i] = 1;
    }

    ParallelLayerWrapper wrapper;
    float* vectorC;

    wrapper.loadMatrix(matrixData, size);
    vectorC = wrapper.forwardMatrix(inputVector);

    for(int i = 0; i < size; i++){
        std::cout << vectorC[i] << " " << i << std::endl;
    }
}