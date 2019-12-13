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


#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef CUDA
#include <cuda_runtime.h>
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


#if defined(_MSC_VER) || defined(_WIN32) || defined(WINDOWS)

#include <time.h>
#include <windows.h>
#define random rand
#define srandom srand

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif


struct spams_timezone
{
   int  tz_minuteswest; /* minutes W of Greenwich */
   int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval *tv, struct spams_timezone *tz)
{
   FILETIME ft;
   unsigned __int64 tmpres = 0;
   static int tzflag = 0;

   if (NULL != tv)
   {
      GetSystemTimeAsFileTime(&ft);

      tmpres |= ft.dwHighDateTime;
      tmpres <<= 32;
      tmpres |= ft.dwLowDateTime;

      tmpres /= 10;  /*convert into microseconds*/
      /*converting file time to unix epoch*/
      tmpres -= DELTA_EPOCH_IN_MICROSECS;
      tv->tv_sec = (long)(tmpres / 1000000UL);
      tv->tv_usec = (long)(tmpres % 1000000UL);
   }

   if (NULL != tz)
   {
      if (!tzflag)
      {
         _tzset();
         tzflag++;
      }
      tz->tz_minuteswest = _timezone / 60;
      tz->tz_dsttime = _daylight;
   }

   return 0;
}

#else
#include <sys/time.h>
#endif


#include "linalg.h"

using namespace std;

/// Class Timer 
class Timer {
   public:
      /// Empty constructor
      Timer();
      /// Destructor
      ~Timer();

      /// start the time
      void inline start() { 
         _running=true;
         gettimeofday(_time1,NULL); };
         /// stop the time
         void inline stop() { 
            gettimeofday(_time2,NULL);
            _running=false;
            _cumul+=  static_cast<double>((_time2->tv_sec - (_time1->tv_sec))*1000000 + _time2->tv_usec-_time1->tv_usec)/1000000.0;
         };
         /// reset the timer
         void inline reset() { _cumul=0;  
            gettimeofday(_time1,NULL); };
            /// print the elapsed time
            void inline printElapsed();
            /// print the elapsed time
            double inline getElapsed() const;

   private:
            struct timeval* _time1;
            struct timeval* _time2;
            bool _running;
            double _cumul;
};

/// Constructor
Timer::Timer() :_running(false) ,_cumul(0) {
   _time1 = (struct timeval*)malloc(sizeof(struct timeval));
   _time2 = (struct timeval*)malloc(sizeof(struct timeval));
};

/// Destructor
Timer::~Timer() {
   free(_time1);
   free(_time2);
}

/// print the elapsed time
inline void Timer::printElapsed() {
   if (_running) {
      gettimeofday(_time2,NULL);
      cout << "Time elapsed : " << _cumul + static_cast<double>((_time2->tv_sec -
               _time1->tv_sec)*1000000 + _time2->tv_usec-_time1->tv_usec)/1000000.0 << endl;
   } else {
      cout << "Time elapsed : " << _cumul << endl;
   }
};

/// print the elapsed time
double inline Timer::getElapsed() const {
   if (_running) {
      gettimeofday(_time2,NULL);
      return _cumul + 
         static_cast<double>((_time2->tv_sec -
                  _time1->tv_sec)*1000000 + _time2->tv_usec-
               _time1->tv_usec)/1000000.0;
   } else {
      return _cumul;
   }
}


#endif
