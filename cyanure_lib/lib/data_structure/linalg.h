/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

/* \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File linalg.h
 * \brief Contains Matrix, Vector classes */

#ifndef LINALG_H
#define LINALG_H

#include <fstream>
#include "../macro.h"
#include "../logging/logger.h"

#undef max
#undef min

template <typename floating_type> 
static inline bool isZero(const floating_type lambda_1) {
   return static_cast<double>(abs<floating_type>(lambda_1)) < 1e-99;
}

template <typename floating_type> 
static inline bool isEqual(const floating_type lambda_1, const floating_type lambda_2) {
   return static_cast<double>(abs<floating_type>(lambda_1-lambda_2)) < 1e-99;
}


template <typename floating_type>
static inline floating_type softThrs(const floating_type x, const floating_type lambda_1) {
   if (x > lambda_1) {
      return x-lambda_1;
   } else if (x < -lambda_1) {
      return x+lambda_1;
   } else {
      return 0;
   }
};

template <typename floating_type>
static inline floating_type fastSoftThrs(const floating_type x, const floating_type lambda_1) {
    return x + floating_type(0.5)*(abs<floating_type>(x-lambda_1) - abs<floating_type>(x+lambda_1));
};


template <typename floating_type>
static inline floating_type hardThrs(const floating_type x, const floating_type lambda_1) {
   return (x > lambda_1 || x < -lambda_1) ? x : 0;
};

template <typename floating_type>
static inline floating_type xlogx(const floating_type x) {
   if (x < -1e-20) {
      return INFINITY;
   } else if (x < 1e-20) {
      return 0;
   } else {
      return x*alt_log<floating_type>(x);
   }
}

template <typename floating_type>
static inline floating_type logexp(const floating_type x) {
   if (x < -30) {
      return 0;
   } else if (x < 30) {
      return alt_log<floating_type>( floating_type(1.0) + exp_alt<floating_type>( x ) );
   } else {
      return x;
   }
}

template <typename floating_type>
static inline floating_type logexp2(const floating_type x) {
   return (x > 0) ? x + log_alt<floating_type>(floating_type(1.0)+ exp_alt<floating_type>(-x)) :
      log( floating_type(1.0) + exp_alt<floating_type>( x ) );
}

template <typename floating_type>
static floating_type solve_binomial(const floating_type a, const floating_type b, const floating_type c) {
   const floating_type delta = b*b-4*a*c;
   return (-b + alt_sqrt<floating_type>(delta))/(2*a); // returns single largest solution, assiming delta > 0;
};

template <typename floating_type>
static floating_type solve_binomial2(const floating_type a, const floating_type b, const floating_type c) {
   const floating_type delta = b*b-4*a*c;
   return (-b - alt_sqrt<floating_type>(delta))/(2*a); // returns single largest solution, assiming delta > 0;
};

#endif
