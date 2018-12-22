#pragma once

#include <core/vec.h>

namespace VecTools {

    double dotProduct(const Vec& left, const Vec& right);

    double distanceLq(double q, const Vec& left, const Vec& right);

    double distanceL2(const Vec& left, const Vec& right);

    double distanceL1(const Vec& left, const Vec& right);

    double lqNorm(double q, const Vec& v);

    double norm(const Vec& v);

}
