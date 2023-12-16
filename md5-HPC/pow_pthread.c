#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#include "openssl/md5.h"

//gcc -fopenmp -lpthread -lcrypto pow_pthread.c -o pow_pthread -lm; ./pow_pthread

#define N_WORKERS 256
#define PACKET_SIZE 1
#define MESSAGE_NUM 2

double timeB;
int flag = 0;
int thread_c;
int packets = 0; // packet id to be processed
int packets_max;

pthread_mutex_t lock;
pthread_mutex_t lock_p;
pthread_barrier_t barrier;

typedef struct My_struct{
    int lenNonceMax;
    int requiredZeros;
    int len;
    unsigned char *nonce;
    MD5_CTX *ctx;
} My_struct;

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

// function checks if there exists a hash with required number of trailing zeros
//      nonce - address of byte array which, appended to the data array, 
//              produces the MD5 hash with required number of trailing zeros
//      ctx - address of the current MD5 structure
//      zeros - required number of trailing zeros in the hash
//      lenNonce - number of additional bytes 
//      lenToAdd - number of not yet defined bytes (recursion)

int HashExists(unsigned char *nonce, MD5_CTX *ctx, int zeros, int lenNonce, int lenToAdd, int thread_n, int packet_n){

    int thread_counter = thread_n;

    MD5_CTX ctxNext;
    int dataNext;
    int ret;

    //int delo = 256 / N_WORKERS;
    int delo = 256 / packets_max;
    //int start = (thread_counter) * delo;
    //int end = (thread_counter+1) * delo;
    int start = (packet_n) * delo;
    int end = (packet_n + 1) * delo;
    if (lenToAdd != lenNonce){
        start = 0;
        end = 256;
    }

    if (lenToAdd > 0){
        for (int i = start; i < end; i++){
            
            if (flag!=0){
                break;
                //pthread_exit(NULL);
            }

            ctxNext = *ctx;
            dataNext = i;
            MD5_Update(&ctxNext, &dataNext, 1);
            ret = HashExists(nonce, &ctxNext, zeros, lenNonce, lenToAdd-1, thread_counter, packet_n);
            
            if (ret){
                nonce[lenNonce-lenToAdd] = dataNext;
                //flag=1;
                //pthread_exit(NULL);
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

void* thread_func(void* argument){

    int thread_counter;

    pthread_mutex_lock(&lock);
    thread_counter = thread_c;
    thread_c++;
    pthread_mutex_unlock(&lock);

    My_struct my_struct = *(My_struct*)argument;
    
    int ret = 0;
    // while the packets are still available to process
    while (packets < packets_max && !flag)
    {
        pthread_mutex_lock(&lock_p);
        int packet_n = packets;
        packets += 1;
        pthread_mutex_unlock(&lock_p);
        ret = HashExists(my_struct.nonce, my_struct.ctx, my_struct.requiredZeros, my_struct.len, my_struct.len, thread_counter, packet_n);
        
        if (ret){
            flag = 1;
            timeB = omp_get_wtime();
            //pthread_exit(NULL);
        }
    }
}

//typedef struct My_struct my_struct;

int main(int argc, char *argv[]) {

    pthread_barrier_init(&barrier, NULL, N_WORKERS);
    pthread_mutex_init(&lock, NULL);
    pthread_mutex_init(&lock_p, NULL);

    unsigned char msg [] =  {1, 2, 3, 4, 194, 170, 210, 13}; // produces hash with 7 trailing zeros
    int lenMsgProvided = 6;     // how many of the above bytes we use as input to the nonce search algorithm
    int lenNonceMax = 4;        // maximal nonce length
    int requiredZeros = 6;      // required number of zeros in the hash

    double start = omp_get_wtime();

    double timeInbetween = 0;
    for (int o = 0; o < MESSAGE_NUM; o++)
    {
        if (o) timeInbetween += omp_get_wtime() - timeB;
        // process MD5 hashing on initial data
        MD5_CTX ctx;
        MD5_Init (&ctx);
        MD5_Update (&ctx, msg, lenMsgProvided);

        unsigned char *nonce = (unsigned char *)malloc (sizeof(unsigned char)*lenNonceMax);

        pthread_t t[N_WORKERS];
        packets_max = (int) ceil((double) 256 / PACKET_SIZE);
        
        double start = omp_get_wtime();

        for (int len = 0; len <= lenNonceMax; len++){

            My_struct my_struct;
            
            my_struct.len = len;
            my_struct.lenNonceMax = lenNonceMax;
            my_struct.requiredZeros = requiredZeros;
            my_struct.nonce = nonce;
            my_struct.ctx = &ctx;

            for (int i = 0; i < N_WORKERS; i++){
                My_struct *a = malloc(sizeof(My_struct));
                *a = my_struct;

                pthread_create(&t[i], NULL, &thread_func, &my_struct);
            }

            for (int i = 0; i < N_WORKERS; i++){
                pthread_join(t[i], NULL);
            }

            if (flag != 0){
                printSolution(msg, lenMsgProvided, nonce, len);
                break;
            }
            thread_c=0;
            packets = 0;
        }
        if (flag == 0)
            printf("Cannot generate the required hash.\n");
        flag = 0;
        free (nonce);
    }
    
    double end = omp_get_wtime();
    printf("Niti: %d \n", N_WORKERS);
    printf("Time: %f seconds\n", (end - start) / MESSAGE_NUM);
    printf("Time before new message: %f\n", timeInbetween / MESSAGE_NUM);

    pthread_mutex_destroy(&lock);
    pthread_mutex_destroy(&lock_p);

    return 0;
}
