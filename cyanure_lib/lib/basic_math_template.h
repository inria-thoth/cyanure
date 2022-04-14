/*!
 * \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File misc.h
 * \brief Contains miscellaneous functions */


#ifndef MISC_H
#define MISC_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "macro.h"



/// a useful debugging function
static inline void stop();
/// seed for random number generation
static int seed = 0;
/// first random number generator from Numerical Recipe
template <typename T> static inline T ran1(); 
/// random sampling from the normal distribution
template <typename T> static inline T normalDistrib();
/// reorganize a sparse table between indices beg and end,

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

static inline void stop() {
   printf("Appuyez sur une touche pour continuer\n");
   getchar();
}

/// first random number generator from Numerical Recipe
template <typename T> static inline T ran1() {
   const int IA=16807,IM=2147483647,IQ=127773,IR=2836,NTAB=32;
   const int NDIV=(1+(IM-1)/NTAB);
   const T EPS=3.0e-16,AM=1.0/IM,RNMX=(1.0-EPS);
   static int iy=0;
   static int iv[NTAB];
   int j,k;
   T temp;

   if (seed <= 0 || !iy) {
      if (-seed < 1) 
        seed=1;
      else seed = -seed;
      for (j=NTAB+7;j>=0;j--) {
         k=seed/IQ;
         seed=IA*(seed-k*IQ)-IR*k;
         if (seed < 0) 
            seed += IM;
         if (j < NTAB) 
            iv[j] = seed;
      }
      iy=iv[0];
   }
   k=seed/IQ;
   seed=IA*(seed-k*IQ)-IR*k;
   if (seed < 0) 
    seed += IM;
   j=iy/NDIV;
   iy=iv[j];
   iv[j] = seed;
   if ((temp=AM*iy) > RNMX) 
    return RNMX;
   else return temp;
};

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

#endif