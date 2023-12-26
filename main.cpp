#include "CudaParallelWrapper.hpp"
#include <cstddef>
#include <iostream>

int main(){
    size_t size = 100;

    CudaParallelWrapper wrapper(size);

    float* vectorA = new float[size];
    for(int i = 0; i < size; i++){
        vectorA[i] = 2;
    }

    float* vectorB = new float[size];
    for(int i = 0; i < size; i++){
        vectorB[i] = 3;
    }

    float vectorC = wrapper.dotVectors(vectorA, vectorB);

    std::cout << vectorC << " " << std::endl;
}