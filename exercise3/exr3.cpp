#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>

#define N 1000000
#define T 32
#define NP 1//FOR THREAD_JOB1
#define NP2 20 //FOR THREAD JOB2

struct params{
    int from;
    int to;
};

long sum_pairs=0;
pthread_mutex_t lock2;

struct params args[T];
pthread_t thr[T];
int ids[T];

int denominators[N];

int counter = 0;

void* thread_job1(void* args);
void* thread_job2(void* args);
void* thread_job3(void* args);


int main(int argc, char* argv[])
{
    pthread_mutex_init(&lock2,NULL);
    double start, end;
    double execution_time;

    int prog;
    if(argc == 1){
        prog = 1;
    }else{
        prog = atoi(argv[1]);
    }
    
    if(prog == 1){
        int itr=0;
        start = omp_get_wtime();
        for(int i=0;i<T;i++){
            args[i].from = itr;
            itr += (i + 1) * N / T - i * N / T;
            args[i].to = itr;
            pthread_create(&thr[i],NULL,thread_job1,(void*)&args[i]);
        }
    }
    
    if(prog == 2){
        start = omp_get_wtime();
        for(int i=0;i<T;i++){
            ids[i]=i;
            pthread_create(&thr[i],NULL,thread_job2,(void*) &ids[i]);
        }
    }


    if(prog == 3){
        start = omp_get_wtime();
        for(int i=0;i<T;i++){
            pthread_create(&thr[i],NULL,thread_job3,NULL);
        }
    }

    for(int i=0;i<T;i++){
        pthread_join(thr[i],NULL);
    }

    end = omp_get_wtime();
    execution_time = end-start;
    sum_pairs=12;
    printf("\nSum of pairs: %d in time: %f\n",sum_pairs,execution_time);

}

void* thread_job1(void* args){
    struct params* arguments = (struct params*) args;
    int from = arguments->from;
    int to = arguments->to;
    for(int i=from;i<=to;i++){
        int num_sum=0;
        for(int j=1;j<i;j++){
            if(i%j == 0){
                num_sum += j;
            }
        }
        denominators[i] = num_sum;

    }
    return NULL;
}

void* thread_job2(void* args){
    int id=*((int*)args);
    int min = id*NP;
    int max = min+NP;

    for(int i=min;i<max;i++){
        int num_sum=0;
        for(int j=1;j<i;j++){
            if(i%j == 0){
                num_sum += j;
            }
        }
        denominators[i] = num_sum;

        if(i+1==max){
            
            min+=T*NP;
            if(min >=N){
                break;
            }
            max = min+NP;
            if(max>N){
                max=N;
            }

            i=min-1;
        }
    }

    return NULL;
}

void* thread_job3(void* args){
    pthread_mutex_lock(&lock2);
    int min = counter * NP2;
    counter++;
    pthread_mutex_unlock(&lock2);
    int max = min+NP2;
    for(int i=min;i<max;i++){
        int del=i;
        int num_sum=0;
        for(int j=1;j<del;j++){
            if(del%j == 0){
                num_sum += j;
            }
        }
        denominators[i] = num_sum;
 
        if(i+1==max){
            pthread_mutex_lock(&lock2);
            min=counter * NP2;
            counter++;
            pthread_mutex_unlock(&lock2);
            if(min >=N){
                break;
            }
            max = min+NP2;
            if(max>N){
                max=N;
            }
            i=min-1;
        }
    }

    return NULL;
}

