#pragma once

#include <core/vec.h>

namespace VecTools {

    double dotProduct(const Vec& left, const Vec& right);

    Vec fill(double alpha, Vec x);

    Vec makeSequence(double from, double step, Vec x);
//    Vecscale(Vecx, double alpha);
//
//    Vecadd(Vecx, const Vec& y);
//
//    Vec sum(const Vec& x, const Vec& y);

    Vec subtract(Vec x, const Vec& y);

//    Vec sub(const Vec& x, const Vec& y);
//
//    Vecadjust(const Vec& x, double alpha);
//
//
//
    Vec pow(double p, const Vec& from, Vec to);
    Vec pow(double p, Vec x);

    Vec sign(const Vec& x, Vec to);
    Vec abs(Vec to);

    Vec mul(const Vec& x, Vec y);
    Vec mul(double alpha, Vec x);

//
//    Veclog(Vecx);
//
//    Vecabs(Vecx);

}
