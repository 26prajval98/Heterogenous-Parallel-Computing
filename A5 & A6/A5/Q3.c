#include "omp.h"
#include<stdio.h>
#include<math.h>
void check(long long int a[],long long int b[], long long int n);
void main()
{
    long long int n;
    printf("Enter the size of vectors: ");
    scanf("%lld",&n);
    long long int i,X[n],Y[n],a;
    for(i=0;i<n;i++)
        X[i] = ((long long int)rand());
    for(i=0;i<n;i++)
        Y[i] = ((long long int)rand());
    printf("Etner the value of a: ");
    scanf("%lld",&a);
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        int ID = omp_get_num_threads();
        #pragma for
        for(i=0;i<n;i++)
        {
            X[i] = a*X[i] + Y[i];
            printf("%lld ID:%d\n",X[i],ID);
        }
    }
    long long int A[n],B[n];
    for(i=0;i<n;i++)
    {
        A[i]=X[i];
        B[i]=Y[i];
    }
    check(A,X,n);
}
void check(long long int a[],long long int b[], long long int n)
{
    long long int i,f=0;
    for(i=0;i<n;i++)
    {
        if(a[i]!=b[i])
        {
            f=1;
            break;
        }
    }
    if(f==1)
        printf("Parallel and sequential do not match");
    else
        printf("Parallel and sequential match!\n");
}