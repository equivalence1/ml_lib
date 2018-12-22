#include <core/vec_tools/distance.h>
#include <core/vec_tools/transform.h>
#include <core/vec_tools/stats.h>

#include <cmath>
#include <cassert>

namespace VecTools {

    double dotProduct(const Vec& left, const Vec& right) {
        assert(left.dim() == right.dim());
        auto result = at::dot(left, right);
        return result.data<float>()[0];
    }

    double distanceLq(double q, const Vec& left, const Vec& right) {
        assert(left.dim() == right.dim());
        return std::pow(sum(abs(left - right) ^ q), 1.0 / q);
    }

    double distanceL2(const Vec& left, const Vec& right) {
        return distanceLq(2, left, right);
    }

    double distanceL1(const Vec& left, const Vec& right) {
        return distanceLq(1, left, right);
    }

    double lqNorm(double q, const Vec& v) {
        return std::pow(sum(absCopy(v) ^ q), 1.0 / q);
    }

    double norm(const Vec& v) {
        return lqNorm(2, v);
    }

}
