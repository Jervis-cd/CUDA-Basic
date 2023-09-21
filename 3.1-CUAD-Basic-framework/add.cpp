#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//定义使用的常量
const double EPSILON=1.0e-15;
const double a=1.23;
const double b=2.34;
const double c=3.57;

//函数声明
void add(const double* x,const double* y,const double* z,const int N);
void check(const double* z,const int N);

//主函数
int main(){

    const int N=100000000;
    const int M=sizeof(double)*N;

    double* x=(double*) malloc(M);
    double* y=(double*) malloc(M);
    double* z=(double*) malloc(M);

    // 初始化数组x,y
    for(int n=0;n<N;++n){

        x[n]=a;
        y[n]=b;
    }

    add(x,y,z,N);
    check(z,N);

    free(x);
    free(y);
    free(z);
    return 0;
}

//函数定义
void add(const double* x,const double* y,const double* z,const int N){

    for(int n=0;n<N;++n){

        z[n]=x[n]+y[n];
    }
}

void check(const double* z,const int N){

    bool has_error=false;
    for(int n=0;n<N;++n){

        if(fabs(z[n]-c)<EPSILON){               //比较两个浮点数的大小必须使用如下方式

            has_error=true;
        }
    }

    printf("%s\n",has_error? "Has errors":"No errors");
}
