#ifndef CONFIGURE_BLAS_H
#define CONFIGURE_BLAS_H

#include <cblas.h>

#ifdef HAVE_MKL
extern "C" {
    void MKL_Set_Num_Threads(int n_threads);
    int MKL_Get_Max_Threads();
}
#endif

#ifdef HAVE_OPENBLAS
extern "C" {
    void openblas_set_num_threads(int n_threads);
}
#endif

static inline void set_mkl_sequential() {
#ifdef HAVE_MKL
    MKL_Set_Num_Threads(1);
#endif
};

static inline void set_mkl_parallel() {
#ifdef HAVE_MKL
#ifdef _OPENMP
    MKL_Set_Num_Threads(omp_get_max_threads());
#endif
#endif
};


static inline int init_omp(const int numThreads) {
    int NUM_THREADS;
#ifdef _OPENMP
    NUM_THREADS = (numThreads == -1) ? MIN(MAX_THREADS, omp_get_num_procs()) : numThreads;
    omp_set_dynamic(1);
    omp_set_num_threads(NUM_THREADS);
    omp_set_max_active_levels(1);
#ifdef HAVE_MKL
    set_mkl_parallel();
#elif defined(HAVE_OPENBLAS)
    openblas_set_num_threads(NUM_THREADS);
#endif
#else
    NUM_THREADS = 1;
#endif
    return NUM_THREADS;
}


#endif