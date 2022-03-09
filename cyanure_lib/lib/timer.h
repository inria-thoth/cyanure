#ifndef TIMER_H
#define TIMER_H


#if defined(_MSC_VER) || defined(_WIN32) || defined(WINDOWS)

#include <time.h>
#include <windows.h>
#include <random>
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> distr(0,INT_MAX);
static int random() {
   return distr(gen);
};
static void srandom(const int seed) {
   gen.seed(seed);
};

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


#include "data_structure/linalg.h"

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
#ifdef CUDA
         (cudaDeviceSynchronize());
#endif
         _running=true;
         gettimeofday(_time1,NULL); };
         /// stop the time
         void inline stop() { 
#ifdef CUDA
         (cudaDeviceSynchronize());
#endif
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
      logging(logINFO) << "Time elapsed : " << _cumul + static_cast<double>((_time2->tv_sec -
               _time1->tv_sec)*1000000 + _time2->tv_usec-_time1->tv_usec)/1000000.0;
   } else {
      logging(logINFO) << "Time elapsed : " << _cumul;
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