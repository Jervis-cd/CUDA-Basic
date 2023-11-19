#include "cuda_tools.hpp"


//定义命名空间CUDATools
namespace CUDATools{

    bool check_runtime(cudaError_t e,const char *call,int line,const char *file){

        if(e!=cudaSuccess){

            INFOE("CUDA Runtime error %s # %s,code=%s [%d] in file %s:%d",call,cudaGetErrorString(e),cudaGetErrorName(e),e,file,line);
            return false;
        }
        retunr true;
    }
}

