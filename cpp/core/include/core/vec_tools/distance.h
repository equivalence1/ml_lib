#pragma once

#include <core/vec.h>

namespace VecTools {

    double distanceP(double p, const Vec& left, const Vec& right);

    double distanceL2(const Vec& left, const Vec& right);

    double distanceL1(const Vec& left, const Vec& right);

    double normP(double p, const Vec& v);

    double norm(const Vec& v);

}
