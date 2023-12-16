//
// Proof of Work
//

// when using #include "openssl/md5.h": compile with: gcc -lcrypto -o pow pow.c
// when using #include "md5.h":         compile with: gcc -o pow md5.c pow.c
// gcc -fopenmp -lcrypto pow_OpenMP.c -o pow_OpenMP -lm; srun --reservation=fri --cpus-per-task={THREADS} ./pow_OpenMP {THREADS}
// gcc -fopenmp -lcrypto pow_OpenMP.c -o pow_OpenMP -lm; ./pow_OpenMP 1

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "openssl/md5.h"

#define MESSAGE_NUM 2

int THREADS;
int solution = -1;
double timeB;

// converts hash of length lenHash to a string, representing hash with hex numbers
//      str - result
//      hash - address of byte array
//      lenHash - number of data bytes
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

// returs number of trailing zeros in string hashStr
int NumberOfTrailingZeros (char *hashStr)
{
    int i = 31;
    while (i >= 0 && hashStr[i--] == '0');
    return 30-i;
}

// print a set of data bytes concatenated with nonce, MF5 hash, and number of trailing zeros
//      msg - address of message array
//      lenMsg - number of message bytes
//      nonce - address of nonce array
//      lenNonce - number of nonce bytes
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
    printf ("nonce len: %d\n", lenNonce);
    
    return 0;
}

// function checks if there exists a hash with required number of trailing zeros
//      nonce - address of byte array which, appended to the data array, 
//              produces the MD5 hash with required number of trailing zeros
//      ctx - address of the current MD5 structure
//      zeros - required number of trailing zeros in the hash
//      lenNonce - number of additional bytes 
//      lenToAdd - number of not yet defined bytes (recursion)
//      threadN - ID of the current thread
int HashExists (unsigned char *nonce, MD5_CTX *ctx, int zeros, int lenNonce, int lenToAdd, int threadN)
{
    MD5_CTX ctxNext;
    int dataNext;
    int ret;

    int start = 0;
    int end = 256;
    if (lenNonce == lenToAdd && threadN != -1)
    {
        start = threadN * 256 / THREADS;
        end = (threadN + 1) * 256 / THREADS;
    }

    if (lenToAdd > 0)
    {
        for (int i = start; i < end; i++)
        {
            if (solution != -1) break;

            ctxNext = *ctx;
            dataNext = i;
            MD5_Update (&ctxNext, &dataNext, 1);
            ret = HashExists (nonce, &ctxNext, zeros, lenNonce, lenToAdd-1, threadN);
            if (threadN == solution)
            {
                nonce[lenNonce-lenToAdd] = dataNext;
                break;
            }
        }
    }
    else
    {
        unsigned char hash [16];
        char hashStr[33];
        ctxNext = *ctx;
        MD5_Final (hash, &ctxNext);
        ValueToString (hashStr, hash, 16);
        ret = NumberOfTrailingZeros (hashStr) == zeros;
        if (ret)
        {
            #pragma omp critical
            solution = threadN;
            timeB = omp_get_wtime();
        }
    }

    return ret;
}

int main (int argc, char *argv[])
{
    unsigned char msg [] =  {1, 2, 3, 4, 194, 170, 210, 13}; // produces hash with 7 trailing zeros
    int lenMsgProvided = 6;     // how many of the above bytes we use as input to the nonce search algorithm
    int lenNonceMax = 4;        // maximal nonce length
    int requiredZeros = 6;      // required number of zeros in the hash

    THREADS = atoi(argv[1]);
    double start = omp_get_wtime();
    omp_set_num_threads(THREADS);
    //THREADS = 256;

    double timeInbetween = 0;
    for (int o = 0; o < MESSAGE_NUM; o++)
    {
        if (o) timeInbetween += omp_get_wtime() - timeB;

        // process MD5 hashing on initial data
        MD5_CTX ctx;
        MD5_Init (&ctx);
        MD5_Update (&ctx, msg, lenMsgProvided);

        // brute-force search for additinal data to get hash with required number of trailing zeros
        unsigned char *nonce = (unsigned char *)malloc (sizeof(unsigned char)*lenNonceMax);

        for (int len = 0; len <= lenNonceMax; len++)
        {
            #pragma omp parallel
            {
                #pragma omp master
                {
                    // Check if the original message produces enough zeros without multithreading
                    if (len == 0) HashExists (nonce, &ctx, requiredZeros, len, len, -1);
                    else
                    {
                        for (int i = 0; i < THREADS - 1; i++)
                        {
                            #pragma omp task shared(solution)
                            HashExists (nonce, &ctx, requiredZeros, len, len, i);
                        }
                        // Master thread also completes a task, while waiting for other threads
                        HashExists (nonce, &ctx, requiredZeros, len, len, THREADS - 1);
                        #pragma omp taskwait
                    }
                }
            }
            if (solution != -1) 
            {
                printSolution(msg, lenMsgProvided, nonce, len);
                break;
            }
        }
        if (solution == -1)
            printf("Cannot generate the required hash.\n");
        free (nonce);
        solution = -1;
    }

    double end = omp_get_wtime();
    //printf("Rezultat: %d \n", rezultat);
    printf("Niti: %d \n", THREADS);
    printf("Time: %f seconds\n", end - start);
    //printf("Pohitritev: %f\n", 254.238083 / (end - start));
    printf("Time before new message: %f\n", timeInbetween / MESSAGE_NUM);
    
    return 0;
}
