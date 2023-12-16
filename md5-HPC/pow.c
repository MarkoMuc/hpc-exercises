//
// Proof of Work
//

// when using #include "openssl/md5.h": compile with: gcc -lcrypto -o ./bin/pow pow.c
// when using #include "md5.h":         compile with: gcc -o pow md5.c pow.c

#include <stdio.h>
#include <stdlib.h>
#include "openssl/md5.h"

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
    
    return 0;
}

// function checks if there exists a hash with required number of trailing zeros
//      nonce - address of byte array which, appended to the data array, 
//              produces the MD5 hash with required number of trailing zeros
//      ctx - address of the current MD5 structure
//      zeros - required number of trailing zeros in the hash
//      lenNonce - number of additional bytes 
//      lenToAdd - number of not yet defined bytes (recursion)
int HashExists (unsigned char *nonce, MD5_CTX *ctx, int zeros, int lenNonce, int lenToAdd)
{
    MD5_CTX ctxNext;
    int dataNext;
    int ret;

    if (lenToAdd > 0)
    {
        for (int i = 0; i < 256; i++)
        {
            ctxNext = *ctx;
            dataNext = i;
            MD5_Update (&ctxNext, &dataNext, 1);    
            ret = HashExists (nonce, &ctxNext, zeros, lenNonce, lenToAdd-1);
            if (ret)
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
    }

    return ret;
}

int main (int argc, char *argv[])
{
    unsigned char msg [] =  {1, 2, 3, 4, 194, 170, 210, 13}; // produces hash with 7 trailing zeros
    int lenMsgProvided = 6;     // how many of the above bytes we use as input to the nonce search algorithm
    int lenNonceMax = 2;        // maximal nonce length
    int requiredZeros = 7;      // required number of zeros in the hash

    // process MD5 hashing on initial data
    MD5_CTX ctx;
    MD5_Init (&ctx);
    MD5_Update (&ctx, msg, lenMsgProvided);

    // brute-force search for additinal data to get hash with required number of trailing zeros
    int ret = 0;
    unsigned char *nonce = (unsigned char *)malloc (sizeof(unsigned char)*lenNonceMax);
    for (int len = 0; len <= lenNonceMax; len++)
    {
        ret = HashExists (nonce, &ctx, requiredZeros, len, len);
        if (ret) 
        {
            printSolution(msg, lenMsgProvided, nonce, len);
            break;
        }
    }
    if (!ret)
        printf("Cannot generate the required hash.\n");
    free (nonce);
    
    return 0;
}
