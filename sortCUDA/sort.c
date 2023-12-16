#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/*
    gcc -fopenmp -o sortCPU sort.c
*/

void fillVector(int *V, int N);
void sort(int* vhod,int* izhod, int N);


int main(int argc, char* argv[])
{
    int length = atoi(argv[1]);

    int* array;
    int* arrayOut;

    array = (int*)malloc(sizeof(int)*length);
    arrayOut = (int*)malloc(sizeof(int)*length);
    fillVector(array,length);
    //for(int i = 0; i < length ; i++) printf("%d ",array[i]);

    double start = omp_get_wtime();
    
    sort(array,arrayOut,length);

    printf("TIME:%03fs FOR LEN:%d\n",omp_get_wtime()-start,length);

    // for(int i = 0; i < length ; i++) printf("%d ",arrayOut[i]);
    // printf("\n");
    free(array);
    free(arrayOut);

}


void sort(int *in, int* out, int N)
{
    int counter;
    for (int i=0; i<N;i++){
        counter=0;
        for (int j=0;j<N;j++){
             
            if(in[i]>in[j] || (in[i]==in[j] && (j<i))){
              counter++;
            }
        } 
        out[counter]=in[i];
    }   
}


void fillVector(int *V, int N)
{
    for(int i = 0; i < N; i++){
        V[i] = rand() % 100;
    }
}
