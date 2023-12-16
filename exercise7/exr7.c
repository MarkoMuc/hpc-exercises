#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "mpi.h"

//TODO: add free(), check if work correctly
/*
    mpicc exr7.c -o exr7

        10 run              4 run               2 run average
   _____|___________________|___________________|_____________
   N\P  1m                  100m                500m
    1   0.4017128           46.0787635
    2   0.2561103           18.2269             104.7695
    4   0.2175525           11.8698             57.2346
    8   0.1999704           9.7025              49.5997   
    16  0.1823689           8.5395              39.3941
    32  0.1911147           6.9036              34.4917

*/

#define N 500000000
#define P 8

void fillVector(int *V);

int cmpfunc (const void * a, const void * b);

int main(int argc, char** argv){
    
    int id;
    int size;
    double start;
    int* sendBuf;
    int* displs;int* scounts;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(id == 0) start = MPI_Wtime();
    
    //Prepare vector
    if(id == 0){
        sendBuf = (int *) malloc(N * sizeof(int));
        fillVector(sendBuf);
    }

    //Prepare displacement and num of elements 
    displs = (int *) malloc(P * sizeof(int));
    scounts = (int *) malloc(P * sizeof(int));

    int offset = 0;

    for(int i = 0; i < size; i++){
        displs[i] = offset; //displacement
        int pi = ((i + 1) * N/P) - (i* N/P);
        offset += pi;
        scounts[i] = offset - displs[i]; //num of elements to send
    }

    //Start Scatter

    int* returnBuf = (int *) malloc(scounts[id] * sizeof(int));

    //Send data
    MPI_Scatterv(sendBuf, scounts, displs, MPI_INT, returnBuf, scounts[id], MPI_INT,0,MPI_COMM_WORLD);

    //Sort data
    qsort(returnBuf,scounts[id],sizeof(int),cmpfunc);


    int* sorted = NULL;

    if(id == 0) sorted = (int *) malloc(N * sizeof(int));

    //Send all data to process 0
    MPI_Gatherv(returnBuf, scounts[id], MPI_INT, sorted, scounts, displs, MPI_INT,0,MPI_COMM_WORLD);

    //final sort and end
    if(id == 0){
        free(sendBuf);
        free(returnBuf);

        //for(int i = 0; i< N ;i++) printf("%d ", sorted[i]);
         
        //qsort(sorted,N,sizeof(int),cmpfunc);
        
       // for(int i = 0; i< N ;i++) printf("%d ", sorted[i]);
        printf("%f\n",MPI_Wtime() - start);
        free(sorted);
        free(scounts);
        free(displs);
    }else{
        free(returnBuf);
        free(scounts);
        free(displs);
    }


    return 0;

}

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

void fillVector(int *V)
{
    for(int i = 0; i < N; i++){
        V[i] = 1 + rand() % 100;
    }
}
