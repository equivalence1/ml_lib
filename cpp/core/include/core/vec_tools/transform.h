#pragma once

#include <core/vec.h>

namespace VecTools {

    Vec copy(const Vec& other);

    Vec copyTo(const Vec& from, Vec to);

    Vec subtract(Vec x, const Vec& y);

    Vec pow(double p, const Vec& from, Vec to);
    Vec pow(double p, Vec x);

    Vec sign(const Vec& x, Vec to);

    Vec abs(Vec to);
    Vec absCopy(const Vec& source);

    Vec mul(const Vec& x, Vec y);
    Vec mul(double alpha, Vec x);
}
