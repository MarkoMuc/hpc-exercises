// compile: gcc -O2 quad.c -fopenmp -lm -o quad

#include <stdio.h>
#include <math.h>
#include <omp.h>

#define TOL 1e-8

double func(double x)
{
    return sin(x*x);
}

double quad(double (*f)(double), double lower, double upper, double tol)
{
    double quad_res;        
    double h;               // intervala len
    double middle;          // intervala mid point
    double quad_coarse;     // coarse aprox
    double quad_fine;       // fine aprox
    double quad_lower;      // result on lower interval  
    double quad_upper;      // result on upper interval
    double eps;             // difference

    h = upper - lower;
    middle = (lower + upper) / 2;

    quad_coarse = h * (f(lower) + f(upper)) / 2.0;
    quad_fine = h/2 * (f(lower) + f(middle)) / 2.0 + h/2 * (f(middle) + f(upper)) / 2.0;
    eps = fabs(quad_coarse - quad_fine);
    
    
    if (eps > tol)
    {
        #pragma omp task shared(f,lower,middle,tol)
        quad_lower = quad(f, lower, middle, tol / 2);
        
        #pragma omp task shared(f,upper,middle,tol)
        quad_upper = quad(f, middle, upper, tol / 2);
        
        quad_res = quad_lower + quad_upper;
    }
    else
        quad_res = quad_fine;

    #pragma omp taskwait
    return quad_res;
}

int main(int argc, char* argv[])
{
    double quadrature;
    double dt = omp_get_wtime();

    #pragma omp parallel
    #pragma omp sin
    quadrature = quad(func, 0.0, 50.0, TOL);

    dt = omp_get_wtime() - dt;

    printf("Integral: %lf\nCas: %lf\n", quadrature, dt);

    return 0;
}
