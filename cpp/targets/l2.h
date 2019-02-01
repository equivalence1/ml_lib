#pragma once

#include "target.h"
#include "stat_based_loss.h"
#include <vec_tools/transform.h>
#include <vec_tools/distance.h>
#include <vec_tools/stats.h>
#include <util/parallel_executor.h>

struct L2Stat {
    using Numeric = double;
    Numeric Sum = 0;
    Numeric Weight = 0;

    L2Stat& operator+=(const L2Stat& other) {
        Sum += other.Sum;
        Weight += other.Weight;
        return *this;
    }

    L2Stat& operator-=(const L2Stat& other) {
        Sum -= other.Sum;
        Weight -= other.Weight;
        return *this;
    }

    void clear() {
        Sum = Weight = 0;
    }
};


inline L2Stat operator-(const L2Stat& left, const L2Stat& right) {
    L2Stat res = left;
    res -= right;
    return res;
}

inline L2Stat operator+(const L2Stat& left, const L2Stat& right) {
    L2Stat res = left;
    res += right;
    return res;
}



class RegularizedL2Score {
public:
    RegularizedL2Score(double lambda = 0)
    : lambda_(lambda) {
        assert(lambda_ >= 0);
        lambda_ += 1e-20;
    }

    double bestIncrement(const L2Stat& stat) const {
        return stat.Weight > 1e-20 ? stat.Sum / (stat.Weight + lambda_) : 0;
    }

    double score(const L2Stat& stat) const {
        return stat.Weight > 1e-20 ? -stat.Sum * stat.Sum / (stat.Weight + lambda_) : 0;
    }
private:
    double lambda_ = 1;
};

class LogL2Score {
public:
    double bestIncrement(const L2Stat& stat) const {
        return stat.Weight > 1 ? stat.Sum / stat.Weight : 0;
    }

    double score(const L2Stat& stat) const {
        return stat.Weight > 1 ? -stat.Sum * stat.Sum * (1 + 2 * log(stat.Weight + 1)) / stat.Weight : 0;
    }
};

//using ScoreFunction = RegularizedL2Score;
using ScoreFunction = LogL2Score;

class L2 :  public Stub<Target, L2>,
            public StatBasedLoss<L2Stat>  {
public:

    explicit L2(const DataSet& ds, Vec target, ScoreFunction scoreFunction = ScoreFunction())
    : Stub<Target, L2>(ds)
    , nzTargets_(target)
    , scoreFunction_(scoreFunction) {

    }

    explicit L2(const DataSet& ds, ScoreFunction scoreFunction = ScoreFunction())
        : Stub<Target, L2>(ds)
          , nzTargets_(ds.target())
          , scoreFunction_(scoreFunction) {

    }


    L2(const DataSet& ds,
       Vec target,
       Vec weights,
       Buffer<int32_t> indices,
       ScoreFunction scoreFunction = ScoreFunction()
       )
        :  Stub<Target, L2>(ds)
        , nzTargets_(std::move(target))
        , nzWeights_(std::move(weights))
        , nzIndices_(std::move(indices))
        , scoreFunction_(scoreFunction) {

    }


    class Der : public Stub<Trans, Der> {
    public:
        Der(const L2& owner)
            :  Stub<Trans, Der>(owner.dim(), owner.dim())
               , owner_(owner) {

        }

        Vec trans(const Vec& x, Vec to) const {
            VecTools::copyTo(owner_.nzTargets_, to);
            to -= x;
            return to;
        }

    private:
        const L2& owner_;
    };

    double bestIncrement(const L2Stat& stat) const  override {
        return scoreFunction_.bestIncrement(stat);
    }


    void makeStats(Buffer<L2Stat>* stats, Buffer<int32_t>* indices) const override {
        (*stats) = Buffer<L2Stat>(nzTargets_.dim());
        if (nzIndices_.size()) {
            (*indices) = nzIndices_.copy();
        } else {
            (*indices) = Buffer<int32_t>(nzTargets_.dim());
            auto indicesRef = indices->arrayRef();
            for (int32_t i = 0; i < indicesRef.size(); ++i) {
                indicesRef[i] = i;
            }
        }

        auto nzTargetsRef = nzTargets_.arrayRef();
        auto nzWeightsRef = nzWeights_.dim() ? nzWeights_.arrayRef() : ConstArrayRef<float>((const float*)nullptr, (size_t)0u);

        ArrayRef<L2Stat> statsRef = stats->arrayRef();
        if (!nzWeightsRef.empty()) {
            parallelFor(0, nzTargetsRef.size(), [&](int64_t i) {
                statsRef[i].Sum = nzTargetsRef[i];
                statsRef[i].Weight = nzWeightsRef[i];
            });
        } else {
            parallelFor(0, nzTargetsRef.size(), [&](int64_t i) {
                statsRef[i].Sum = nzTargetsRef[i];
                statsRef[i].Weight = 1.0;
            });
        }
    }


    double score(const L2Stat& comb) const override {
        return scoreFunction_.score(comb);
    }

    DoubleRef valueTo(const Vec& x, DoubleRef to) const {
        assert(nzWeights_.dim() == 0);
        assert(nzIndices_.size() == 0);
        to = VecTools::sum((x - nzTargets_) ^ 2);
        to /= x.dim();
        to = sqrt(to);
        return to;
    }

private:
    //TODO(noxoomo): this should be just sparse vec instead
    Vec nzTargets_;
    Vec nzWeights_;
    Buffer<int32_t> nzIndices_;
    ScoreFunction scoreFunction_;

};


