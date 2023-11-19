# include "cuda_tools.hpp"
# include <stdio.h>

int main(int argc,char * argv[]){

    int device_id=0;
    if (argc>1) device_id=atoi(argv[1]);
    checkCudaRuntime(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    checkCudaRuntime(cudaGetDeviceProperties(&(prop,device_id)));

    printf("Device id:",device_id);
    printf("Device name:",prop.name);
    printf("Compute capability:",prop.major,prop.minor);
    printf("Amount of global memory:",prop.totalGlobalMem/(1024*1024*1024.0));
    printf("Amount of constant memory:",prop.totalConstMem/1024);
}