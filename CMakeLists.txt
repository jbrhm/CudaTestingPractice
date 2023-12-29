cmake_minimum_required( VERSION 3.2 )

project( CudaTesting LANGUAGES CXX CUDA )

find_package( CUDA REQUIRED )

set( FILES main.cpp CudaParallelWrapper.hpp CudaParallelWrapper.cu CudaParallel.cuh CudaParallel.cu ParallelLayer.cu ParallelLayer.cuh ParallelLayerWrapper.cu ParallelLayerWrapper.cuh )

add_executable( CudaTesting ${FILES} )