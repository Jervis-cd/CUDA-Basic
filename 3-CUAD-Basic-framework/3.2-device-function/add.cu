#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//定义使用的常量
const double EPSILON=1.0e-15;
const double a=1.23;
const double b=2.34;
const double c=3.57;

//函数声明
__global__ void add(const double* x,const double* y,double* z,const int N);
void check(const double* z,const int N);

//主函数
int main(){

    const int N=100000000;
    const int M=sizeof(double)*N;

    //在host端分配内存
    double *h_x=(double*) malloc(M);
    double *h_y=(double*) malloc(M);
    double *h_z=(double*) malloc(M);

    // 在host初始化数组x,y
    for(int n=0;n<N;++n){

        h_x[n]=a;
        h_y[n]=b;
    }

    double *d_x,*d_y,*d_z;
    //device端分配内存,(void **)强制内存转换，将某种类型的双重指针转化为void类型的双重指针,此处为: double ** ---> void **
    cudaMalloc((void **)&d_x,M);
    cudaMalloc((void **)&d_y,M);
    cudaMalloc((void **)&d_z,M);

    //复制host端数据到device端
    cudaMemcpy(d_x,h_x,M,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,M,cudaMemcpyHostToDevice);

    //调用核函数
    const int block_size=128;
    const int grid_size=N/block_size;
    add<<<grid_size,block_size>>>(d_x,d_y,d_z,N);

    cudaError_t cudaStatus = cudaGetLastError();

    //将计算好的数据从device到host
    cudaMemcpy(h_z,d_z,M,cudaMemcpyDeviceToHost);
    check(h_z,N);

    //释放内存
    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;
}

//三种版本不会带来性能差异
//版本一:有返回值的设备函数
__device__ double add1_device(const double x,const double y){

    return (x+y);
}

//版本二:用指针传参的设备函数
__device__ void add2_device(const double x,const double y,double *z){

    *z=x+y;                 //对z解引用后赋值
}

//版本三:用引用传参的设备函数
__device__ void add3_device(const double x,const double y,double &z){

    z=x+y;                 //对z解引用后赋值
}

//核函数定义
__global__ void add(const double* x,const double* y,double* z,const int N){

    //计算线程索引
    const int n=blockDim.x*blockIdx.x+threadIdx.x;

    //设置if语句，规避不需要的线程操作，造成内存的非法访问
    if(n<N){
        z[n]=add1_device(x[n],y[n]);        //版本一设备函数调用
        add2_device(x[n],y[n],&z[n]);       //版本二设备函数调用
        add3_device(x[n],y[n],z[n]);        //版本三设备函数调用
    }
}

void check(const double* z,const int N){

    bool has_error=false;
    for(int n=0;n<N;++n){

        if(fabs(z[n]-c)>EPSILON){               //判断两个浮点是否相等必须使用如下方式
            has_error=true;
        }
    }

    printf("%s\n",has_error? "Has errors":"No errors");
}
