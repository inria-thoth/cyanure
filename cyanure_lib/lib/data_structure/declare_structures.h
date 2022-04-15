#ifndef DECLARE_STRUCTURE_H
#define DECLARE_STRUCTURE_H

#include "../basic_math_template.h"
#include "../BLAS/cblas_alt_template.h"
#include "../BLAS/configure_blas.h"

/// Sparse Vector class
template<typename floating_type, typename I = INTM> class SpVector;

/// Dense Matrix class
template<typename floating_type> class Matrix;

/// Sparse Matrix class
template<typename floating_type, typename I = INTM> class SpMatrix;

/// Dense Vector class
template<typename floating_type> class Vector;

/// Dense OptimInfo class
template<typename floating_type> class OptimInfo;


#endif