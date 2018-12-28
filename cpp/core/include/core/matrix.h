#pragma once

#include "vec.h"
#include "scalar.h"

enum class MatrixLayout {
    RowMajor
};

class Mx {
public:
    Mx(Mx&& other) = default;

    Mx(Mx& other) = default;

    Mx(Vec& x, int64_t nrows, int64_t ncols, MatrixLayout layout = MatrixLayout::RowMajor)
        : vec_(x)
        , layout_(layout)
        , rows_(nrows)
        , cols_(ncols) {
    }

    Mx(const Vec& x,
       int64_t nrows,
       int64_t ncols,
       MatrixLayout layout = MatrixLayout::RowMajor
       )
    : vec_(x)
    , layout_(layout)
    , rows_(nrows)
    , cols_(ncols) {
        assert(vec_.dim() == rows_ * cols_);
    }

    Mx(int64_t nrows, int64_t ncols, MatrixLayout layout = MatrixLayout::RowMajor)
    : vec_(nrows * ncols)
    , layout_(layout)
    , rows_(nrows)
    , cols_(ncols) {
        assert(vec_.dim() == rows_ * cols_);
    }


    Mx& set(int64_t x, int64_t y, Scalar val);

    Scalar get(int64_t x, int64_t y) const;

    operator Vec() {
        return  vec_;
    }

    operator Vec() const {
        return  vec_;
    }

    Mx& operator+=(const Mx& other);
    Mx& operator-=(const Mx& other);


    Mx& operator+=(Scalar value);
    Mx& operator-=(Scalar value);
    Mx& operator*=(Scalar value);
    Mx& operator/=(Scalar value);


    Mx& operator^=(Scalar q);
private:
    int64_t seqIndex(int64_t x, int64_t y) const;
private:
    Vec vec_;
    MatrixLayout layout_ = MatrixLayout::RowMajor;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
};
