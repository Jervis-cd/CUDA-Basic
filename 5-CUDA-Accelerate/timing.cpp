#include <cuda.h>
#include <cuda_runtime.h>

// CUDA事件类型
CUDAEvent_t start,stop;

// 使用cudaEventCreate()函数初始化CUDA事件变量
checkCudaRuntime(cudaEventCreate(&start));
checkCudaRuntime(cudaEventCreate(&stop));

// 在需要计时的代码之前记录一个代表开始的事件
checkCudaRuntime(cudaEventRecord(&(start)));

/* 
对于TCC驱动模式的GPU可以省略，但是处于WDDM驱动模式的GPU必须保留
因为处于WDDM驱动模式的GPU中，一个CUDA流中的操作并不是直接交给GPU执行
而是先提交到一个软件队列，需要添加一条对流的CUDAEventQuery操作（cudaEventSynchronize）刷新队列，
才能促使前面的操作在GPU中执行
*/
cudaEventQuery(start);               //此处不需要使用checkCudaRuntime函数，见上一章


// 需要计时的代码,可以是主机代码，也可时设备代码


// 记录一个代表事件结束的事件
checkCudaRuntime(cudaEventRecord(stop));
// 让主机等待事件stop被记录完毕
checkCudaRuntime(cudaEventSynchronize(stop));
// 计算start和stop两个事件的时间差
float elapsed_time;
checkCudaRuntime(cudaEventElapsedTime(&elapsed_time,start,stop));
printf("Time=%g ms.\n",elapsed_time);
// 销毁start和stop事件
checkCudaRuntime(cudaEventDestroy(start));
checkCudaRuntime(cudaEventDestroy(stop));
