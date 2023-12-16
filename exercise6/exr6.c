#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include "mpi.h"

#define N 6
#define TAG_DATA 2
#define TAG_SIZE 4

int main(int argc, char** argv){
    
    int id;
    int size;
    int bufferSize;
    char* buffer;
    char* outMessage;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    

    //0 send 1 start message 
    if(id == 0){
        char* out = "0";
        bufferSize = strlen(out)+1;
        
        int sendTo = size == 1 ? 0 : id+1;

        MPI_Send(&bufferSize,1,MPI_INT,sendTo,TAG_SIZE,MPI_COMM_WORLD);
        MPI_Send((void*) out,bufferSize,MPI_CHAR,sendTo,TAG_DATA,MPI_COMM_WORLD);
    }
    
    //WAIT & receive
    int waitFor = id !=0 ? id-1 : size-1;
    MPI_Recv(&bufferSize,1, MPI_INT, waitFor, TAG_SIZE, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    buffer = (char*) malloc(sizeof(char) * bufferSize);
    MPI_Recv(buffer,bufferSize, MPI_CHAR, waitFor, TAG_DATA, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    
    //BUILD STRING
    int outSize = snprintf(outMessage,0,"%s - %d ",buffer,id);
    outMessage = (char*) malloc(sizeof(char) * outSize+1);
    sprintf(outMessage,"%s - %d",buffer,id);
    int sendTo = id == size-1? 0 : id+1;
    
    //SEND OR PRINT MESSAGE
    if(id != 0){
        bufferSize = strlen(outMessage)+1;
        MPI_Send(&bufferSize,1,MPI_INT,sendTo,TAG_SIZE,MPI_COMM_WORLD);
        
        MPI_Send((void*)outMessage,bufferSize,MPI_CHAR,sendTo,TAG_DATA,MPI_COMM_WORLD);
    }else{
        printf("%s\n",outMessage);
    }
    free(buffer);
    MPI_Finalize();
    
    return 0;
}
