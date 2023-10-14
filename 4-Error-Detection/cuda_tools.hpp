#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 版本一,官方使用方式，使用宏函数形式
// call表示CUDA运行时API函数
#define checkRuntime(call)                                                     \
do{                                                                             \
    const cudaError_t error_code=call;                                          \
    if(error_code != cudaSuccess){                                              \
                                                                                \
        printf("CUDA Error:\n");                                                \
        printf("File %s\n",__FILE__);                                           \
        printf("Line %s\n",__LINE__);                                           \
        printf("Error code: %d\n",error_code);                                  \
        printf("Error text:%s\n",cudaGetErrorString(error_code));               \
        exit(1);                                                                \ 
    }                                                                           \
}while(0)

// 版本二，使用普通函数进行CUDA运行时函数检查
// call表示CUDA运行时API函数，#call表示取API函数名字的字符串
#define checkCudaRuntime(call) CUDATools::check_runtime(call,#call,__LINE__,__FILE__)
