#pragma once

#include <core/vec.h>

namespace VecTools {

    Scalar dotProduct(const Vec& left, const Vec& right);

    Scalar distanceLq(Scalar q, const Vec& left, const Vec& right);

    Scalar distanceL2(const Vec& left, const Vec& right);

    Scalar distanceL1(const Vec& left, const Vec& right);

    Scalar lqNorm(Scalar q, const Vec& v);

    Scalar norm(const Vec& v);

}
