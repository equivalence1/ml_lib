#include "cross_entropy.h"
#include <vec_tools/transform.h>
#include <vec_tools/stats.h>


inline Vec sigmoid(const Vec& point) {
    Vec x = VecTools::expCopy(point);
    auto tmp = x + 1.0;
    x /= tmp;
    return x;
}

inline void crossEntropyGradient(const Vec& target, const Vec& point, Vec to) {
    Vec p = sigmoid(point);
    VecTools::copyTo(target, to);
    to -= p;
}

void CrossEntropy::subsetDer(const Vec& point, const Buffer<int32_t>& indices, Vec to) const {
    Vec gatheredPoint(indices.size());
    Vec gatheredTarget(indices.size());
    VecTools::gather(point, indices, gatheredPoint);
    VecTools::gather(target_, indices, gatheredTarget);
    crossEntropyGradient(gatheredTarget, gatheredPoint, to);
}

Vec CrossEntropy::gradientTo(const Vec& x, Vec to) const {
    crossEntropyGradient(target_, x, to);
    return to;
}

DoubleRef CrossEntropy::valueTo(const Vec& x, DoubleRef to) const {
    auto tmp = VecTools::expCopy(x);

    // t * log(p(x)) + (1.0 - t) * log(1.0 - p(x));
    // t log(s(x)) + (1.0 - t) * log(1.0 - s(x))
    //t log(s(x)) + (1.0 - t)  * log(s(-x))
    //t (x - log(1 + exp(x)) + (1.0 - t) * (-log(1.0 + exp(x))
    //t * x - log(1.0 + exp(x))
    to = VecTools::sum(target_ * x - VecTools::log(tmp + 1.0));
    return to;
}

CrossEntropy::CrossEntropy(const DataSet& ds,
                           const Vec& target,
                           const Vec& weights)
    : Stub(ds)
    , target_(target)
    , weights_(weights) {
    totalWeight_ = VecTools::sum(weights_);

}
