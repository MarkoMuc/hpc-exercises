#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"
#include "md5.h"

/*
mpicc md5.c md5_mpi.c -o ./bin/md5_mpi
*/
#define N_WORKERS 18

void ValueToString (char *str, unsigned char *hash, int lenHash)
{
    char sb[3];

    for (int i=0; i<lenHash; i++)
    {
        sprintf (sb, "%02x", hash[i]);
        str[2*i] = sb[0];
        str[2*i+1] = sb[1];
    }
    str[2*lenHash] = 0;
}

int NumberOfTrailingZeros (char *hashStr)
{
    int i = 31;
    while (i >= 0 && hashStr[i--] == '0');
    return 30-i;
}

int printSolution(unsigned char *msg, int lenMsg, unsigned char *nonce, int lenNonce)
{    
    MD5_CTX ctx;
    unsigned char hash [16];
    char hashstr[33];

    MD5_Init (&ctx);
    MD5_Update (&ctx, msg, lenMsg);
    MD5_Update (&ctx, nonce, lenNonce);
    MD5_Final (hash, &ctx);
    ValueToString (hashstr, hash, 16);

    printf ("data : ");
    for (int i=0; i<lenMsg; i++)
        printf ("%02x", msg[i]);    
    for (int i=0; i<lenNonce; i++)
        printf ("%02x", nonce[i]);
    printf ("\nhash : %s\n", hashstr);
    printf ("zeros: %d\n", NumberOfTrailingZeros (hashstr));
    
    return 0;
}

#define TAG_FINISH 8

int checkIfFinished(MPI_Request* req,int* finished, MPI_Status* status,int id){
    int flagTest = 0;
   MPI_Test(req,&flagTest,status);
    if(flagTest == 1){
       return 1;
    }else{
        return 0;
    }

}

int HashExists(unsigned char *nonce, MD5_CTX *ctx, int zeros, int lenNonce, int lenToAdd, int id,MPI_Request* req,int* finished,MPI_Status* status){
    MD5_CTX ctxNext;
    int dataNext;
    int ret;

    int delo = 256 / (N_WORKERS-3);
    int start = (id-3) * delo;
    int end = (id-2) * delo;
    if (lenToAdd != lenNonce){
        start = 0;
        end = 256;
    }

    if (lenToAdd > 0){
        for (int i = start; i < end; i++){
            int finish = 0;
            if(ret != -1) finish = checkIfFinished(req,finished,status,id);    

            if(finish == 1){
                return -1;
            }

            ctxNext = *ctx;
            dataNext = i;
            MD5_Update(&ctxNext, &dataNext, 1);
            ret = HashExists(nonce, &ctxNext, zeros, lenNonce, lenToAdd-1, id,req,finished,status);

            if (ret){
                nonce[lenNonce-lenToAdd] = dataNext;
                break;
            }
        }
    } else {
        unsigned char hash[16];
        char hashStr[33];
        ctxNext = *ctx;
        MD5_Final(hash, &ctxNext);
        ValueToString(hashStr, hash, 16);
        ret = NumberOfTrailingZeros (hashStr) == zeros;
    }
    return ret;

}


//TAGS FOR COMMUNICATION
#define TAG_RESULT 1
#define TAG_STOP 2
#define TAG_DATA 3
#define TAG_START 4
#define TAG_MSG 5
#define TAG_MSG_SIZE 6
#define TAG_NONCE 7
#define TAG_MSG_CLIENT 9
#define TAG_DATA_CLIENT 10
#define TAG_STOP_CLIENT 11
#define TAG_NOUNCE_CLIENT 12
#define TAG_RESULT_CLIENT 13
#define TAG_LEN_CLIENT 14

#define DATA_SIZE 4


int main(int argc, char *argv[]) {

    //Setup MPI
    int id;
    int size;
    double startTime;
    int stop = 0; //signal
    int start = 0; //signal
    int ret;
    double startSend;
    double endSend;
    int count = 0;

    MPI_Status status;
    MPI_Request req;

    //Buffers

    //len : 0
    //lenNonceMax : 1
    //requiredZeros : 2
    //lenMessageProvided : 3
    int* data;
    int clients[2];
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    //START
    /*
    unsigned char msg [] =  {1, 2, 3, 4, 194, 170, 210, 13}; // produces hash with 7 trailing zeros
    int lenMsgProvided = 6;     // how many of the above bytes we use as input to the nonce search algorithm
    int lenNonceMax = 2;        // maximal nonce length
    int requiredZeros = 7;      // required number of zeros in the hash
    if(id == 0) startTime = MPI_Wtime();
    */

    if(id == 0) startTime = MPI_Wtime();
    
    
    int requiredZeros;
    int lenMsgProvided;
    int lenNonceMax;
    unsigned char* msg;

    int messageCount;
    unsigned char *nonce;

    data = (int*) malloc(sizeof(int)*DATA_SIZE);
    if(size < 4){
        MPI_Abort(MPI_COMM_WORLD,-4);
    }
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
    clients[0] = 0;
    clients[1] = 0;

    //ID == 0 aka ROOT, receives message, sends work to workers
    while(id == 0){
        int client = -1;
        int nounceReturnLen = -1;

        //TO CHECK IF ANY MESSAGES ARE LEFT
        int receivedAll = 0;

        //receives messageCount from client
        MPI_Recv(&messageCount,1,MPI_INT,MPI_ANY_SOURCE,TAG_STOP_CLIENT,MPI_COMM_WORLD,&status);
        client = status.MPI_SOURCE;

        if(messageCount == 0){
            //THERE IS NO MESSAGES LEFT ON THIS CLIENT
            printf("ROOT:received all from client: %d\n",client);
            fflush(stdout);

            clients[client - 1] = -1;
        }else{
            //THERE ARE STILL MESSAGES LEFT ON THIS CLIENT
            clients[client - 1] = 0;
        }

        //CHECKS IF ALL MESSAGES HAVE BEEN RECEIVED
        for(int i = 0; i < 2;i++){
            if(clients[i] == -1){
                receivedAll++;
            }
        }

        //received no data from client, so go to next iteration 
        if(messageCount == 0 && receivedAll != 2){
            continue;
        }

        if(receivedAll != 2){
            if(count == 1){
                endSend = MPI_Wtime();
                count++;
            }
            //ROOT RECEIVES DATA PACKAGE FROM CLIENT
            MPI_Recv(data,DATA_SIZE,MPI_INT,client,TAG_DATA_CLIENT,MPI_COMM_WORLD,&status);
           //data[3] = lenMessageProvided
            msg = (unsigned char *)malloc (sizeof(unsigned char)*data[3]);
            
            //ROOT RECEIVES MSG PACKAGE FROM CLIENT
            MPI_Recv(msg,data[3],MPI_UNSIGNED_CHAR,client,TAG_MSG_CLIENT,MPI_COMM_WORLD,&status);
        }


        //checks if nounce is not needed ->if true, continue;
        if(messageCount != 0){
            MD5_CTX ctx;
            MD5_Init (&ctx);
            MD5_Update (&ctx, msg, data[3]);
            ret = 0;
            int finished = 0;
            unsigned char hash[16];
            char hashStr[33];
            MD5_Final(hash, &ctx);
            ValueToString(hashStr, hash, 16);
            ret = NumberOfTrailingZeros (hashStr) == requiredZeros;
            if(ret == 1){
                //printf("found\n");
                int len = -1;
                MPI_Send(&ret,1,MPI_INT,client,TAG_RESULT_CLIENT,MPI_COMM_WORLD);
                MPI_Send(&len,1,MPI_INT,client,TAG_LEN_CLIENT,MPI_COMM_WORLD);
                free(msg);
                continue;
            }
        }
        
        //Tells workers if more work will be sent
        for(int worker=3;worker < N_WORKERS && receivedAll == 2;worker++){
            
            //printf("ROOT: sending STOP kill signal to workers\n");
            stop = 1; // sends "KILL" to workers
            MPI_Send(&stop,1,MPI_INT,worker,TAG_STOP,MPI_COMM_WORLD);
        }

        //End ROOT after sending "KILL" to workers
        if(receivedAll == 2){
            break;
        }
        
        //data[1] = lenNonceMax
        nonce = (unsigned char *)malloc (sizeof(unsigned char)*data[1]);
        
        //Sends all data to workers
        for(int len = 0; len <= data[1];len++){
            int finished = 0;
            int startWorker = 1;
            stop = 0;
            for (int worker = 3; worker < N_WORKERS; worker++){
                data[0] = len;
                /*
                data[1] = lenNonceMax;
                data[2] = requiredZeros;
                data[3] = lenMsgProvided;
                */
                MPI_Send(&stop,1,MPI_INT,worker,TAG_STOP,MPI_COMM_WORLD);
                
                MPI_Send(&startWorker,1,MPI_INT,worker,TAG_START,MPI_COMM_WORLD);
                
                MPI_Send(data,DATA_SIZE,MPI_INT,worker,TAG_DATA,MPI_COMM_WORLD);
                
                MPI_Send(msg,data[3],MPI_UNSIGNED_CHAR,worker,TAG_MSG,MPI_COMM_WORLD);
            }

            int skipMe = 0;
            //listens, for workers who have finished, if their ret == 1 breaks, otherwise wait for all to end
            while(finished < N_WORKERS - 3){
            
                MPI_Recv(&ret,1,MPI_INT,MPI_ANY_SOURCE,TAG_RESULT,MPI_COMM_WORLD,&status);
                  
                if(ret == 1){
                    if(id == 0 && count == 0){
                        startSend = MPI_Wtime();
                        count ++;
                    } 
                    //receives NOUNCE from worker that sent ret == 1
                    skipMe = status.MPI_SOURCE;
                    //printf("ROOT: receiving nounce from:%d on len:%d\n",skipMe,data[0]);
                    MPI_Recv(nonce, data[1],MPI_UNSIGNED_CHAR,skipMe,TAG_NONCE,MPI_COMM_WORLD,&status);
                    finished++;
                    //TODO:
                    break;
                }else{
                    finished++;
                }
            }
            //Finished calculating for this len

            //Stop workers, if all of them havent finished yet
            if(finished != N_WORKERS - 3){
                //TODO: add check for finish  in hashCheck()
                //printf("finishing\n");
                int finished = 1;
                for(int i = 3; i < N_WORKERS; i++){
                    if(i == skipMe){
                        continue;
                    }
                    MPI_Send(&finished,1,MPI_INT,i,TAG_FINISH,MPI_COMM_WORLD);
                }
            }

            //if found right answer, break and dont go and calc for larger len
            if(ret == 1){
                nounceReturnLen = len;
                break;
            }
        }

        //SENDS return back to CLIENT
        MPI_Send(&ret,1,MPI_INT,client,TAG_RESULT_CLIENT,MPI_COMM_WORLD);

        if(ret == 1){
            if(nounceReturnLen == -1){
                //Sets to nounceLenMax, if it wasnt done earlier
                nounceReturnLen = data[1];
            }
            MPI_Send(&nounceReturnLen,1,MPI_INT,client,TAG_LEN_CLIENT,MPI_COMM_WORLD);
            MPI_Send(nonce,data[1],MPI_UNSIGNED_CHAR,client,TAG_NOUNCE_CLIENT,MPI_COMM_WORLD);
        }

        //FREE MEMORY
        free(msg);
        free(nonce);
    }
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
    int finished = 0;
    int flagTest = 0;
    //Workers: wait for data, calculates, sends back to ROOT

    while(id > 2 ){

        MD5_CTX ctx;
        finished = 0;
        flagTest = 0;
        ret = 0;
        if(start == 0){
            //MPI_Barrier(MPI_COMM_WORLD);
            //printf("%d WORKER: waiting\n",id);
            //fflush(stdout);

            MPI_Recv(&stop,1,MPI_INT,0,TAG_STOP,MPI_COMM_WORLD,&status);
            if(stop == 1){
                //FINISHED ALL WORK
                //printf("%d WORKER: ENDING\n",id);
                //fflush(stdout);
                break;
            }
            MPI_Recv(&start,1,MPI_INT,0,TAG_START,MPI_COMM_WORLD,&status);    
            MPI_Recv(data,DATA_SIZE,MPI_INT,0,TAG_DATA,MPI_COMM_WORLD,&status);
            //mem recv and malloc
            msg = (unsigned char *)malloc (sizeof(unsigned char)*data[3]);
          
            MPI_Recv(msg,data[3],MPI_UNSIGNED_CHAR,0,TAG_MSG,MPI_COMM_WORLD,&status);
            
            //nonce malloc
            nonce = (unsigned char *)malloc (sizeof(unsigned char)*data[1]);
            
            //Calcs ctx
            MD5_Init (&ctx);
            MD5_Update (&ctx, msg, data[3]);
        }

        //Calculate
        MPI_Irecv(&finished,1,MPI_INT,0,TAG_FINISH,MPI_COMM_WORLD,&req);
        ret = HashExists(nonce,&ctx,data[2],data[0],data[0],id,&req,&finished,&status);
       // printf("%d WORKER: finished calculating return:%d\n",id,ret);
        //fflush(stdout);

        //DONT SEND RESULTS IF RECEIVED FINISH(if ret == -1)!
        if(ret == -1){
            start = 0;
            //printf("%d WORKER: received FINISH\n",id);
            continue;
        }else{
            //printf("%d WORKER:finshed with no premature finish\n",id);
            //fflush(stdout);
            MPI_Cancel(&req);
            MPI_Request_free(&req);
        }

        // int pomozni = MPI_Test_cancelled(&req);
        // if(pomozni == 1){
        //     start =
        // }

        //send result

        MPI_Send(&ret,1,MPI_INT,0,TAG_RESULT,MPI_COMM_WORLD);
    
        //send nounce if got correct answer
        if(ret == 1){
            printf("%d WORKER: sending nonce\n",id);
            fflush(stdout);
            MPI_Send(nonce,data[1],MPI_UNSIGNED_CHAR,0,TAG_NONCE,MPI_COMM_WORLD);
        }

        //FINISHED work,reset
        if(start == 1){
            start = 0; //waits for more work
        }
        free(msg);
        free(nonce);

    }

    if( id == 0){
        printf("%f\n",MPI_Wtime() - startTime);
        printf("send time: %f\n",endSend-startSend);
    }
//----------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
    //IF YOU WANT DIFF DATA IN BOTH OF THEM; JUST CHANGE THE IF and mor e msgs
    unsigned char msg2 [] =  {1, 2, 3, 4, 194, 170, 210, 13};
    //CLIENTS SET UP DATA
    if(id == 1){
        lenMsgProvided = 6;
        lenNonceMax = 2;  
        requiredZeros = 7;
        messageCount = 1;
    }

    //CLIENTS SET UP DATA 
    if(id == 2){
        lenMsgProvided = 6;
        lenNonceMax = 2;  
        requiredZeros = 7;
        messageCount = 1;
    }

    while(id > 0 && id < 3 && messageCount >=0){
 
        //ADD READ FROM FILE/ ANY OTHER GENERATE MSG; ALSO ADD MSG MALLOC HERE


        int len = 0;
        //Sends how many messages are left to root/coordinator, if its <1 stop listening from me!
        MPI_Send(&messageCount,1,MPI_INT,0,TAG_STOP_CLIENT,MPI_COMM_WORLD);
        
        if(messageCount == 0){
            printf("%d CLIENT: done sending all messages\n",id);
            fflush(stdout);
            break;
        }

        //Sends data to root, if it has any left
        if(messageCount != 0){
            data[0] = -1;
            data[1] = lenNonceMax;
            data[2] = requiredZeros;
            data[3] = lenMsgProvided;
            MPI_Send(data,DATA_SIZE,MPI_INT,0,TAG_DATA_CLIENT,MPI_COMM_WORLD);
            MPI_Send(msg2,lenMsgProvided,MPI_UNSIGNED_CHAR,0,TAG_MSG_CLIENT,MPI_COMM_WORLD);
        }

        //Waits to receive the data
        MPI_Recv(&ret,1,MPI_INT,0,TAG_RESULT_CLIENT,MPI_COMM_WORLD,&status);
        
        if(ret == 1){
            printf("%d CLIENT: Results\n",id);
            MPI_Recv(&len,1,MPI_INT,0,TAG_LEN_CLIENT,MPI_COMM_WORLD,&status);

            if(len != -1){
                nonce = (unsigned char *)malloc (sizeof(unsigned char)*lenNonceMax);
                MPI_Recv(nonce,lenNonceMax,MPI_UNSIGNED_CHAR,0,TAG_NOUNCE_CLIENT,MPI_COMM_WORLD,&status);
                printSolution(msg2,lenMsgProvided,nonce,len);
                free(nonce);
            }else if( len == -1){
                nonce = (unsigned char *)malloc (sizeof(unsigned char)*1);
                printSolution(msg2,lenMsgProvided,nonce,0);
                free(nonce);
            }
        }else{
            printf("%d CLIENT: DID NOT FIND NONCE\n",id);
        }
        
        //ADD FREE MSG IF YOU WANT DIFF MSGs
        //free(msg2)

        messageCount--;

    }


    MPI_Finalize();

    free(data);
    return 0;
}
