#include "exr4.h"

/*
    COMPILE : 
    gcc -o exr4 exr4.c -pthread -fopenmp -lm -O2

*/


int main(int argc, char* argv[])
{
    double start;
    double end;

    height = (int) (ceil(log(T)/log(2)));
    height = height == 0 ? 1:height;

    //PREPARE AND FILL: VECTOR AND BARRIERS
    A = (double*) malloc(N * sizeof(double));
    B = (double*) malloc(N * sizeof(double));
    barriers = (pthread_barrier_t*) malloc((height+1) * sizeof(pthread_barrier_t));
    int barrierC=T;
    for(int i=0;i<=height;i++){
        pthread_barrier_init(&barriers[i],NULL,barrierC);
        barrierC = barrierC/2;
        barrierC = barrierC == 0 ? 1 : barrierC;
    }


    fillVector(A);
    fillVector(B);

    //START CLOCK AND RUN THREADS
    start = omp_get_wtime();
    
    int past=0;
    int ti=0;
    for(int i=0;i<T;i++){
        ti+=(i + 1) * N / T - i * N / T;
        pars[i].bound1 = past;
        pars[i].bound2 = ti;
        pars[i].id = i;
        pthread_create(&threads[i],NULL,thread_job,(void*)&pars[i]);
        past = ti;
    }

    for(int i=0;i<T;i++){
        pthread_join(threads[i],NULL);
    }
    results[0] = sqrt(results[0]);//Thread 1
    end = omp_get_wtime();
    
    printf("Rez:%f TIME: %f\n",results[0],end-start);
    

    //CLEAR MEM AND EXIT
    free(A);
    free(B);
    free(barriers);
    
    return 0;
}

void* thread_job(void* arg){
    

    //GET PARAMETERS
    struct params* args = (struct params*) arg;
    int bound1 = args->bound1;
    int bound2 = args->bound2;
    int id= args->id;

    //CALC Distance
    double result = pdistance(bound1,bound2);
    results[id] = result;

    pthread_barrier_wait(&barriers[0]);
    
    //COLLAPSE 
    id++;
    

    int triplet = 0;

    int t_left = T;
    int offset = 1;
    triplet = t_left % 2 == 1 ? t_left : 0;

   
    for(int i =0;i<height;i++){

        if(id%(2*offset) !=1  || triplet == id){
            
            //printf("T:%dDIED on height %d\n",id,i);
            //fflush(stdout);
            
            return NULL;
        }

        results[id-1] += results[id+offset-1];
    
        if(t_left % 2 == 1 && triplet-2*offset == id){
            results[id-1] += results[triplet-1];
        }
        
        pthread_barrier_wait(&barriers[i+1]);
        offset *= 2;
        t_left /=2;
        triplet = triplet != 0 ? (triplet-offset):0;
    }
    return NULL;
}



double pdistance(int i,int n){
    
    double result = 0.0;
    for (i; i < n; i++)
    {
        double rez = (A[i]- B[i]);
        result += rez * rez;
    }
    return result;
    
}

void fillVector(double *V)
{
    for(int i = 0; i < N; i++){
        V[i] = rand() / (1.0 * RAND_MAX);
    }
}
