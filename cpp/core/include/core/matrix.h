#pragma once

#include <vec.h>

class Matrix : public Vec {
public:
    Matrix(Vec& x, int64_t nrows, int64_t ncols);
    Matrix(const Vec& x, int64_t nrows, int64_t ncols);

    Matrix sub(int64_t i, int64_t j, int64_t height, int64_t weight) const;

    Matrix sub(int64_t i, int64_t j, int64_t height, int64_t weight);

    Vec row(int64_t i);
    Vec col(int64_t i);

    Vec row(int64_t i) const;
    Vec col(int64_t i) const;

    Matrix& set(int64_t x, int64_t y, double val);
    double get(int64_t x, int64_t y) const;
};
