#include <core/scalar.h>
#include <core/vec_tools/distance.h>
#include <core/vec_tools/transform.h>
#include <core/vec_tools/stats.h>

namespace VecTools {

    Scalar dotProduct(const Vec& left, const Vec& right) {
        assert(left.dim() == right.dim());
        return Scalar(at::dot(left, right));
    }

    Scalar distanceLq(Scalar q, const Vec& left, const Vec& right) {
        assert(left.dim() == right.dim());
        return std::pow(sum(abs(left - right) ^ q), 1.0 / q);
    }

    Scalar distanceL2(const Vec& left, const Vec& right) {
        return distanceLq(2, left, right);
    }

    Scalar distanceL1(const Vec& left, const Vec& right) {
        return distanceLq(1, left, right);
    }

    Scalar lqNorm(Scalar q, const Vec& v) {
        return std::pow(sum(absCopy(v) ^ q), 1.0 / q);
    }

    Scalar norm(const Vec& v) {
        return lqNorm(2, v);
    }

}
