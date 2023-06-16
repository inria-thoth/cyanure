#ifndef MULTI_LOGISTIC_LOSS_H
#define MULTI_LOGISTIC_LOSS_H

#include "../loss_mat.h"

template <typename M>
class MultiClassLogisticLoss final: public LinearLossMat<M, Vector<int> > {
    typedef typename M::value_type T;
    using LinearLossMat<M, Vector<int> >::_data;
    using LinearLossMat<M, Vector<int> >::_y;
public:
    MultiClassLogisticLoss(DataMatrixLinear<M>& data, const Vector<int>& y): LinearLossMat<M, Vector<int> >(data, y) {
        _nclasses = y.maxval() + 1;
        this->_id = MULTI_LOGISTIC;
    };

    inline T eval(const Matrix<T>& input) const {
         Matrix<T> tmp;
         _data.pred(input,tmp); 
         const int n = tmp.n();
         T sum=0;
#pragma omp parallel for reduction(+:sum) schedule(static)
         for (int ii = 0; ii<n; ++ii) {
            Vector<T> col;
            tmp.refCol(ii,col);
            col.add(-col[_y[ii]]);
            sum+=col.logsumexp();
         }
         return sum / n;
      };
    inline T eval(const Matrix<T>& input, const INTM i) const {
        Vector<T> tmp;
        _data.pred(i, input, tmp);
        tmp.add(-tmp[_y[i]]);
        return tmp.logsumexp();
    }
    inline void print() const {
        logging(logINFO) << "Multiclass logistic Loss is used";
    };
    inline T fenchel(const Matrix<T>& input) const {
        T sum = 0;
        const int n = input.n();
        //pragma omp parallel for reduction(+:sum) schedule(static) 
        for (long long i = 0; i < n; ++i) {
            const long long clas = _y[i];
            for (long long j = 0; j < _nclasses; ++j) {
                if (j == clas) {
                    sum += xlogx(input[i * _nclasses + j] + 1.0);
                }
                else {
                    sum += xlogx(input[i * _nclasses + j]);
                }
            }
        }
        return sum / n;
    };

private:
    int _nclasses;
    inline void get_grad_aux2(Vector<T>& col, const int ind) const {
        col.add(-col[ind]);
        const T mm = col.maxval();
        col.add(-mm);
        col.exp();
        col.scal(T(1.0) / col.asum());
        col[ind] = 0;
        col[ind] = -col.asum();
    }
    inline void get_grad_aux(const Matrix<T>& input, Matrix<T>& grad1) const {
        _data.pred(input, grad1);
        const int n = grad1.n();
        //pragma omp parallel for schedule(static,16)
        for (int ii = 0; ii < n; ++ii) {
            Vector<T> col;
            grad1.refCol(ii, col);
            get_grad_aux2(col, _y[ii]);
        }
    };
    inline void scal_grad(const Matrix<T>& input, const INTM i, Vector<T>& col) const {
        _data.pred(i, input, col);
        get_grad_aux2(col, _y[i]);
    };
    inline T lipschitz_constant() const { return T(0.25); };
    inline void get_dual_constraints(Matrix<T>& grad1) const {
        // scale grad1 by 1/Nclasses
        if (_data.intercept()) {
            Vector<T> row;
            for (int i = 0; i < grad1.m(); ++i) {
                grad1.extractRow(i, row);
                row.project_sft(_y, i);
                grad1.setRow(i, row);
            }
        }
    }
};

#endif