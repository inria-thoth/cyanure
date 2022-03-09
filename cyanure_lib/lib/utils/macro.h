/*!
 * \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File utils.h
 * \brief Contains various variables and class timer */


#ifndef MACRO_H
#define MACRO_H

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <limits>
#include "../logging/logger.h"


#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MATLAB_MEX_FILE
typedef int mwSize;
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

// MIN, MAX macros
#define MIN(a,b) (((a) > (b)) ? (b) : (a))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define SIGN(a) (((a) < 0) ? -1.0 : 1.0)
#define ABS(a) (((a) < 0) ? -(a) : (a))
// DEBUG macros
#define PRINT_I(name) printf(#name " : %d\n",name);
#define PRINT_F(name) printf(#name " : %g\n",name);
#define PRINT_S(name) printf("%s\n",name);
#define FLAG(a) printf("flag : %d \n",a);

// ALGORITHM constants
#define EPSILON 10e-10
#ifndef INFINITY
#define INFINITY 10e20
#endif
#define EPSILON_OMEGA 0.001
#define TOL_CGRAD 10e-6
#define MAX_ITER_CGRAD 40

#endif
