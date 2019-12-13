/* Copyright (c) 2019, Julien Mairal 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

#ifndef MISC_H
#define MISC_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

#if defined(_MSC_VER) || defined(_WIN32) || defined(WINDOWS)
#define isnan _isnan
#define isinf !_finite
#endif

using namespace std;


/// a useful debugging function
static inline void stop();
/// standard random number generator 
template <typename T> static inline T ran1(); 
/// random sampling from the normal distribution
template <typename T> static inline T normalDistrib();
/// reorganize a sparse table between indices beg and end,
/// using quicksort
template <typename T, typename I>
static void sort(I* irOut, T* prOut,I beg, I end);
template <typename T, typename I>
static void quick_sort(I* irOut, T* prOut,const I beg, const I end, const bool incr);
/// template version of the power function
template <typename T>
T power(const T x, const T y);
/// template version of the fabs function
template <typename T>
T abs(const T x);
/// template version of the fabs function
template <typename T>
T sqr(const T x);
template <typename T>
T sqr_alt(const T x);
/// template version of the fabs function
template <typename T>
T sqr(const int x) {
   return sqr<T>(static_cast<T>(x));
}

template <typename T>
T exp_alt(const T x);
template <typename T>
T log_alt(const T x);

/// a useful debugging function
/*static inline void stop() {
   cout << "Appuyez sur entrée pour continuer...";
   cin.ignore( numeric_limits<streamsize>::max(), '\n' );
};*/
static inline void stop() {
   printf("Appuyez sur une touche pour continuer\n");
   getchar();
}

/// standard random number generator 
template <typename T> T ran1() {
   return static_cast<T>(rand())/RAND_MAX;
}

/// random sampling from the normal distribution
template <typename T>
static inline T normalDistrib() {
   static bool iset = true;
   static T gset;

   T fac,rsq,v1,v2;
   if (iset) {
      do {
         v1 = 2.0*ran1<T>()-1.0;
         v2 = 2.0*ran1<T>()-1.0;
         rsq = v1*v1+v2*v2;
      } while (rsq >= 1.0 || rsq == 0.0);
      fac = sqrt(-2.0*log(rsq)/rsq);
      gset = v1*fac;
      iset = false;
      return v2*fac;
   } else {
      iset = true;
      return gset;
   }
};

/// reorganize a sparse table between indices beg and end,
/// using quicksort
template <typename T, typename I>
static void sort(I* irOut, T* prOut,I beg, I end) {
   I i;
   if (end <= beg) return;
   I pivot=beg;
   for (i = beg+1; i<=end; ++i) {
      if (irOut[i] < irOut[pivot]) {
         if (i == pivot+1) {
            I tmp = irOut[i];
            T tmpd = prOut[i];
            irOut[i]=irOut[pivot];
            prOut[i]=prOut[pivot];
            irOut[pivot]=tmp;
            prOut[pivot]=tmpd;
         } else {
            I tmp = irOut[pivot+1];
            T tmpd = prOut[pivot+1];
            irOut[pivot+1]=irOut[pivot];
            prOut[pivot+1]=prOut[pivot];
            irOut[pivot]=irOut[i];
            prOut[pivot]=prOut[i];
            irOut[i]=tmp;
            prOut[i]=tmpd;
         }
         ++pivot;
      }
   }
   sort(irOut,prOut,beg,pivot-1);
   sort(irOut,prOut,pivot+1,end);
}
template <typename T, typename I>
static void quick_sort(I* irOut, T* prOut,const I beg, const I end, const bool incr) {
   if (end <= beg) return;
   I pivot=beg;
   if (incr) {
      const T val_pivot=prOut[pivot];
      const I key_pivot=irOut[pivot];
      for (I i = beg+1; i<=end; ++i) {
         if (prOut[i] < val_pivot) {
            prOut[pivot]=prOut[i];
            irOut[pivot]=irOut[i];
            prOut[i]=prOut[++pivot];
            irOut[i]=irOut[pivot];
            prOut[pivot]=val_pivot;
            irOut[pivot]=key_pivot;
         } 
      }
   } else {
      const T val_pivot=prOut[pivot];
      const I key_pivot=irOut[pivot];
      for (I i = beg+1; i<=end; ++i) {
         if (prOut[i] > val_pivot) {
            prOut[pivot]=prOut[i];
            irOut[pivot]=irOut[i];
            prOut[i]=prOut[++pivot];
            irOut[i]=irOut[pivot];
            prOut[pivot]=val_pivot;
            irOut[pivot]=key_pivot;
         } 
      }
   }
   quick_sort(irOut,prOut,beg,pivot-1,incr);
   quick_sort(irOut,prOut,pivot+1,end,incr);
}

template <typename T, typename I>
static void quick_sort(T* prOut,const I beg, const I end, const bool incr) {
   if (end <= beg) return;
   I pivot=beg;
   if (incr) {
      const T val_pivot=prOut[pivot];
      for (I i = beg+1; i<=end; ++i) {
         if (prOut[i] < val_pivot) {
            prOut[pivot]=prOut[i];
            prOut[i]=prOut[++pivot];
            prOut[pivot]=val_pivot;
         } 
      }
   } else {
      const T val_pivot=prOut[pivot];
      for (I i = beg+1; i<=end; ++i) {
         if (prOut[i] > val_pivot) {
            prOut[pivot]=prOut[i];
            prOut[i]=prOut[++pivot];
            prOut[pivot]=val_pivot;
         } 
      }
   }
   quick_sort(prOut,beg,pivot-1,incr);
   quick_sort(prOut,pivot+1,end,incr);
}


/// template version of the power function
template <>
inline double power(const double x, const double y) {
   return pow(x,y);
};
template <>
inline float power(const float x, const float y) {
   return powf(x,y);
};

/// template version of the fabs function
template <>
inline double abs(const double x) {
   return fabs(x);
};
template <>
inline float abs(const float x) {
   return fabsf(x);
};

/// template version of the fabs function
template <>
inline double sqr(const double x) {
   return sqrt(x);
};
template <>
inline float sqr(const float x) {
   return sqrtf(x);
};

template <>
inline double exp_alt(const double x) {
   return exp(x);
};
template <>
inline float exp_alt(const float x) {
   return expf(x);
};

template <>
inline double log_alt(const double x) {
   return log(x);
};
template <>
inline float log_alt(const float x) {
   return logf(x);
};


template <>
inline double sqr_alt(const double x) {
   return sqrt(x);
};
template <>
inline float sqr_alt(const float x) {
   return sqrtf(x);
};

#ifdef HAVE_MKL
extern "C" {
   void MKL_Set_Num_Threads(int nthreads);
   int MKL_Get_Max_Threads();
}
#endif

static inline void set_mkl_sequential() {
#ifdef HAVE_MKL
   MKL_Set_Num_Threads(1);
#endif
};

static inline void set_mkl_parallel() {
#ifdef HAVE_MKL
   MKL_Set_Num_Threads(omp_get_max_threads());
#endif
};


static inline int init_omp(const int numThreads) {
   int NUM_THREADS;
#ifdef _OPENMP
   NUM_THREADS = (numThreads == -1) ? MIN(MAX_THREADS,omp_get_num_procs()) : numThreads;
   omp_set_nested(0);
   omp_set_dynamic(0);
   omp_set_num_threads(NUM_THREADS);
   set_mkl_parallel();
#else
   NUM_THREADS = 1;
#endif
   return NUM_THREADS;
}


#endif
