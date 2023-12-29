#pragma once

#include <stddef.h>

#include "ParallelLayerWrapper.cuh"
#include "ParallelLayer.cuh"

ParallelLayerWrapper::ParallelLayerWrapper(){
    m_ParallelLayer = new ParallelLayer();
}

void ParallelLayerWrapper::loadMatrix(float* matrixData, size_t size){
    m_ParallelLayer->loadMatrix(matrixData, size);
}

float* ParallelLayerWrapper::forwardMatrix(float* vectorData){
    return m_ParallelLayer->forwardMatrix(vectorData);
}