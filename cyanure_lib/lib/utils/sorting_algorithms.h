#ifndef SORTING_ALG_H
#define SORTING_ALG_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

using namespace std;


/// using quicksort
template <typename T, typename I>
static void sort(I* irOut, T* prOut,I beg, I end);
template <typename T, typename I>
static void quick_sort(I* irOut, T* prOut,const I beg, const I end, const bool incr);
/// template version of the power function

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
};

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
};

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
};

#endif