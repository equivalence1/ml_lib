#pragma once

#include "vec.h"


class Matrix {
public:
    Matrix(VecRef& x, int64_t nrows, int64_t ncols)
    : data_(x)
    , rows_(nrows)
    , cols_(ncols) {

    }

    Matrix(ConstVecRef& x, int64_t nrows, int64_t ncols)
        : data_(x)
          , rows_(nrows)
          , cols_(ncols) {

    }
//
////    Matrix sub(int64_t i, int64_t j, int64_t height, int64_t weight) const;
////    Matrix sub(int64_t i, int64_t j, int64_t height, int64_t weight);
//
    Vec row(int64_t i);
    Vec col(int64_t i);
//
//
//    Matrix& set(int64_t x, int64_t y, Double val);
//    double get(int64_t x, int64_t y) const;


    operator std::reference_wrapper<Vec>() {
        return std::reference_wrapper<Vec>(*this);
    }
private:
    Vec data_;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
};
