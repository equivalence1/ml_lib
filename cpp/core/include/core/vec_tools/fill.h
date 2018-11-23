#pragma once

#include <core/vec.h>

namespace VecTools {

    double dotProduct(const Vec& left, const Vec& right);

    Vec& fill(double alpha, Vec& x);

    Vec& makeSequence(double from, double step, Vec& x);
//    Vec& scale(Vec& x, double alpha);
//
//    Vec& add(Vec& x, const Vec& y);
//
//    Vec sum(const Vec& x, const Vec& y);

    Vec& subtract(Vec& x, const Vec& y);

//    Vec sub(const Vec& x, const Vec& y);
//
//    Vec& adjust(const Vec& x, double alpha);
//
//
//
    Vec& exp(double p, const Vec& from, Vec& to);

    Vec& mul(Vec& x, const Vec& y);

//
//    Vec& log(Vec& x);
//
//    Vec& abs(Vec& x);

}
