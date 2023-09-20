# CUDA编程基础与实践

## 1. GPU硬件与CUDA程序开发工具

### 1.1 GPU硬件简介

GPU是英文缩写graphics processing unit，意为图像处理器，对应概念CPU为central proccessing unit的缩写，意为中央处理器

CPU和GPU的区别：

* CPU只拥有少数的快速计算核心，而GPU拥有几百到几千个不那么快速的计算核心
* CPU中更多的晶体管用于数据缓存和流程控制，GPU中更多晶体管用于算术逻辑单元

*GPU靠众多的计算核心来获得相对较高的计算性能*

GPU计算不是只单独的GPU计算，而是指CPU+GPU的异构计算，GPU必须在CPU的调度下才能完成特定任务。CPU和GPU分别称之为主机（host）和设备（device），通过PCIe总线进行连接

* FLOPS: 浮点数运算峰值，即每秒最多能执行的浮点数运算次数，表征计算性能的一个重要参数

* GPU内存带宽: 影响计算性能的另一个参数

* 显存: 显存容量也会制约应用程序的性能

### 1.2 CUDA程序开发工具

可以用于GPU编程的几种软件开发工具

* CUAD是Nvidia推出的GPU编程语言
  * CUDA提供两层API，分别为CUDA Driver API（底层）和CUDA Runtime API
  * 应用程序使用GPU：1.调用CUDA库；2.CUDA Runtime API；3.CUDA Driver API，其实最终都是通过CUDA Driver API调用GPU
  * 不同的GPU架构由不同的计算能力，一般由X.Y表征硬件架构的计算能力

* OpenCL是一个更为通用的为各种异构平台编写并行程序的框架，也是AMD的GPU的主要程序开发工具

* OpenACC是由多个公司共同开发的异构并行编程标准

### 1.3 使用命令检查与设置设备

CUDA常用设置命令

```bash
#查看显卡信息
nvidia-smi

#设置调用GPU编号
export CUDA_VISIBLE_DEVICES=id

#设置GPU的计算模式
sudo nvidia-smi -i GPU_ID -c 0	#默认模式
sudo nvidia-smi -i GPU_ID -c 1  #独占进程模式
```



## 2. CUDA中的线程组织

### 2.1 CUDA程序

CUDA程序编译器为nvcc工具，nvcc是C++编译器的超集

CUDA程序由主机代码和设备代码两部分组成，因为GPU的调用需要CPU的参与。主机通过核函数方式对设备进行调用

```C++
// 一个简单，典型的CUDA程序结构
int main(){
  
  主机代码
  核函数调用
  主机代码
  return 0;
}
```

### 2.2 核函数简介

CUDA中的核函数与C++函数类似，显著差别是核函数必须被限定词`__global__`修饰，并且核函数的返回值类型必须为空`void`，`void`和`__global__`的顺序可以交换

```C++
//第一种方式，常用
__global__ void function_name(){}

//第二种方式
void __global__ function_name(){}
```

主机调用核函数时需要制定设备中调用线程个数

核函数中线程的组织为若干线程块（thread block），全部线程块构成一个网格（grid），每个线程块含有相同数目的线程，数目为线程块的大小

```C++
//核函数调用方式
function_name<<<grid_size,block_size>>>();
```

调用核函数之后需要执行

```C++
cudaDeviceSynchronize();			//同步主机和设备
```

调用输出函数（printf）时，输出流时先存放在缓冲区，缓冲区不会自动刷新，只有程序遇到同步操作时缓存区才会刷新

### 2.3 CUAD中的线程组织

一个GPU一般有几千个计算核心，总线程数必须至少等于计算核心数才有可能充分利用GPU中的全部计算资源。实际上，总线程数大于计算核心时才能更充分地利用GPU中的计算资源，因为这会让计算和内存访问之间以及不同的计算之间合理的重叠，减小计算核心空闲时间

#### 2.3.1线程索引

```C++
function_name<<<grid_size,block_size>>>();
//grid_size表示网格大小最多2^31-1，block_size最大为1024
```

`grid_size`和`block_size`两个值分别保存在`gridDim`和`blockDim`两个内建变量中

另外，内建变量`blockIdx` : 线程在网格中的线程块索引；内建变量 `threadIdx`: 线程在线程块中的线程索引

四个内建变量都是多维的，可以通过`grid_size.x`方式访问各个维度，一维时，只有x维度可以访问

计算当前线程在所在线程块的ID：

thread_id=threadIdx.z\*blockDim.x\*blockDim.y+threadIdx.y\*blockDim.x+threadIdx.x

计算当前线程块在grid中的ID:

block_id=blockIdx.z\*gridDim.x\*gridDim.y+blockIdx.y\*gridDim.x+blockIdx.x

一个线程块中的线程还可以细分为不同的线程束。一个线程线程束是同一个线程块中相邻的warpSize个线程。warpSize也是内建变量，表示线程束的大小

#### 2.3.2 网格和线程块大小限制

grid_size三个维度不能大于2^31和65535和65535

block_size三个维度不能大于1024,1024,64，并且线程块的总大小不能超过1024线程

### 2.4 nvcc编译CUDA程序

CUDA的编译器驱动nvcc先将全部源代码分离为主机代码和设备代码。主机代码完整的支持C++语法，设备代码只部分支持C++语法。nvcc先将设备代码编译为PTX伪汇编代码，再将PTX代码编译为二进制的cubin目标代码。

* 将源代码编译为PTX代码时，需要选用-arch=compute_XY指定一个虚拟架构计算能力，用以确定代码中能够使用的CUAD功能。
* 将PTX代码编译为cubin代码时，需要用-code=sm_ZW指定一个真实架构的计算能力，用以确定可执行文件能够使用的GPU。真实架构的计算能力必须等于或大于虚拟架构的计算能力

```bash
-arch=compute_35 -code=sm_60
```

使用-code=sm_ZW指定GPU真实架构为Z.W。对应的可执行文件只能在主版本号为Z，次版本号大于或等于W的GPU中运行

如果希望编译出的可执行文件能够在更多的GPU中执行，可以同时指定多组计算能力

```bash
-gencode arch=compute_35,code=sm_35
-gencode arch=compute_60,code=sm_60
...
```

这样生成可执行文件称为胖二进制文件，因为包含了指定数量的二进制版本，在不同的架构中GPU运行时自动选择对应二进制版本。过多地指定计算能力，会增加编译时间和可执行文件的大小

另外，nvcc有一种机制称为即使编译，在可执行文件中保留一个如下的PTX代码

```bash
-gencode arch=compute_XY,code=compute_XY
```

简化的命令

```bash
-arch=sm_XY
```

上面代码等价于

```bash
-gencode arch=compute_XY,code=sm_ZW
-gencode arch=compute_XY,code=compute_XY
```

也可以不指定计算能力，此时编译时将采用默认计算能力（和CUDA版本相关）

## 3. CUDA程序的基本框架

### 3.1 CUDA程序的基本框架

```C++
头文件包含
常量定义（或宏定义）
C++自定义函数和CUDA核函数的声明
int main(){
  
  分配主机内存与设备内存
  初始化主机中的数据
  将某些数据从主机复制到设备
  调用核函数在设备中进行计算
  将某些数据从设备复制到主机
  释放主机与设备内存
}

C++自定义函数和核函数的定义
```



