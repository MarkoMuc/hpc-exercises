#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>


/*

    Run average of 10 runs where N=100 000 000
    Time Log in ./log/<threadNum>.out


    THREADS |   time[s] |
        1   | 0.2276206 |
        2   | 0.244565  |
        4   | 0.1349059 |     
        8   | 0.1049509 |
        16  | 0.2033105 |
        32  | 0.2334488 |

*/


//MACROS
#define T   4  //thread count
#define N   100000000 //vector size

//STRUCTS
struct params{
    int id;
    int bound1;
    int bound2;
};

//ARRAYS
pthread_t threads[T];
struct params pars[T];
double results[T];

//POINTERS
pthread_barrier_t* barriers;
double* A;
double* B;

//GLOBAL VARS
int height;


//FUNCTIONS
void fillVector(double* V);
double pdistance(int i,int n);
void* thread_job(void* arg);
