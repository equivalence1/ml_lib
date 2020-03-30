#include "linear_l2_stat.h"

namespace {
    [[nodiscard]] inline Eigen::MatrixXd DiagMx(int dim, double v) {
        Eigen::MatrixXd mx(dim, dim);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                if (j == i) {
                    mx(i, j) = v;
                } else {
                    mx(i, j) = 0;
                }
            }
        }
        return mx;
    }
}


// LinearL2CorStat


LinearL2CorStat::LinearL2CorStat(int size)
        : size_(size) {
    xxt.resize(size_, 0.);
    xy = 0.;
}

LinearL2CorStat& LinearL2CorStat::appendImpl(const LinearL2CorStat &other,
                                             const LinearL2CorStatOpParams &opParams) {
    for (int i = 0; i < size_; ++i) {
        xxt[i] += other.xxt[i];
    }
    xy += other.xy;
}

LinearL2CorStat& LinearL2CorStat::removeImpl(const LinearL2CorStat &other,
                                             const LinearL2CorStatOpParams &opParams) {
    for (int i = 0; i < size_; ++i) {
        xxt[i] -= other.xxt[i];
    }
    xy -= other.xy;
}

LinearL2CorStat& LinearL2CorStat::appendImpl(const float* x, float y, float weight,
                                             const LinearL2CorStatOpParams &opParams) {
    const float wf = weight * opParams.fVal;
    for (int i = 0; i < size_ - 1; ++i) {
        xxt[i] += x[i] * wf;
    }
    xxt[size_ - 1] += opParams.fVal * wf;
    xy += y * weight;
}

LinearL2CorStat& LinearL2CorStat::removeImpl(const float* x, float y, float weight,
                                             const LinearL2CorStatOpParams &opParams) {
    const float wf = weight * opParams.fVal;
    for (int i = 0; i < size_ - 1; ++i) {
        xxt[i] -= x[i] * wf;
    }
    xxt[size_ - 1] -= opParams.fVal * wf;
    xy -= y * weight;
}


// LinearL2Stat


LinearL2Stat::LinearL2Stat(int size, int filledSize)
        : size_(size)
        , filledSize_(filledSize) {
    maxUpdatedPos_ = filledSize_;

    w_ = 0;
    sumY_ = 0;
    sumY2_ = 0;
    trace_ = 0;
    xtx_.resize(size * (size + 1) / 2);
    xty_.resize(size);
    sumX_.resize(size);
}

void LinearL2Stat::reset() {
    w_ = 0;
    sumY_ = 0;
    sumY2_ = 0;
    trace_ = 0;
    memset(xtx_.data(), 0, (maxUpdatedPos_ * (maxUpdatedPos_ + 1) / 2) * sizeof(float));
    memset(xty_.data(), 0, maxUpdatedPos_ * sizeof(float));

    filledSize_ = 0;
    maxUpdatedPos_ = 0;
    memset(sumX_.data(), 0, maxUpdatedPos_ * sizeof(float));
}

void LinearL2Stat::addNewCorrelation(const float* xtx, float xty, float w, int shift)  {
    // don't use w for new correlations, it should be already calculated
    // TODO add an option
    (void)sizeof(w);

    const int corPos = filledSize_ + shift;

    int pos = corPos * (corPos + 1) / 2;
    for (int i = 0; i <= corPos; ++i) {
        xtx_[pos + i] += xtx[i];
    }
    xty_[corPos] += xty;
    maxUpdatedPos_ = std::max(maxUpdatedPos_, corPos + 1);
}

void LinearL2Stat::addFullCorrelation(const float* x, float y, float w) {
    w_ += w;
    float yw = y * w;
    sumY_ += yw;
    sumY2_ += yw * y;

    int pos = 0;
    for (int i = 0; i < filledSize_; ++i) {
        float xiw = x[i] * w;
        for (int j = 0; j < i + 1; ++j) {
            xtx_[pos + j] += xiw * x[j];
        }
        sumX_[i] += xiw;
        xty_[i] += xiw * y;
        pos += i + 1;
    }
}

LinearL2Stat& LinearL2Stat::appendImpl(const LinearL2Stat &other, const LinearL2StatOpParams &opParams) {
    w_ += other.w_;
    sumY_ += other.sumY_;
    sumY2_ += other.sumY2_;
    trace_ += other.trace_;

    int size = opParams.opSize;
    if (size < 0) {
        size = std::min(filledSize_, other.filledSize_);
    }

    int pos = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < i + 1; ++j) {
            xtx_[pos + j] += other.xtx_[pos + j];
        }
        pos += i + 1;
        xty_[i] += other.xty_[i];
        sumX_[i] += other.sumX_[i];
    }

    return *this;
}

LinearL2Stat& LinearL2Stat::removeImpl(const LinearL2Stat &other, const LinearL2StatOpParams &opParams) {
    w_ -= other.w_;
    sumY_ -= other.sumY_;
    sumY2_ -= other.sumY2_;
    trace_ -= other.trace_;

    int size = opParams.opSize;
    if (size < 0) {
        size = std::min(filledSize_, other.filledSize_);
    }

    int pos = 0;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < i + 1; ++j) {
            xtx_[pos + j] -= other.xtx_[pos + j];
        }
        pos += i + 1;
        xty_[i] -= other.xty_[i];
        sumX_[i] -= other.sumX_[i];
    }

    return *this;
}

LinearL2Stat& LinearL2Stat::appendImpl(const float* x, float y, float weight,
                                       const LinearL2StatOpParams &opParams) {
    switch (opParams.vecAddMode) {
        case LinearL2StatOpParams::FullCorrelation:
            addFullCorrelation(x, y, weight);
            break;

        case LinearL2StatOpParams::NewCorrelation:
        default:
            addNewCorrelation(x, y, weight, opParams.shift);
            break;
    }

    return *this;
}

LinearL2Stat& LinearL2Stat::removeImpl(const float* x, float y, float weight,
                                       const LinearL2StatOpParams &opParams) {
    appendImpl(x, y, -1 * weight, opParams);
}

void LinearL2Stat::fillXTX(LinearL2Stat::EMx &XTX) const {
    int basePos = 0;
    for (int i = 0; i < maxUpdatedPos_; ++i) {
        for (int j = 0; j < i + 1; ++j) {
            XTX(i, j) = xtx_[basePos + j];
            XTX(j, i) = xtx_[basePos + j];
        }
        basePos += i + 1;
    }
}

LinearL2Stat::EMx LinearL2Stat::getXTX() const {
    EMx res(maxUpdatedPos_, maxUpdatedPos_);
    fillXTX(res);
    return res;
}

void LinearL2Stat::fillXTy(EMx &XTy) const {
    for (int i = 0; i < maxUpdatedPos_; ++i) {
        XTy(i, 0) = xty_[i];
    }
}

LinearL2Stat::EMx LinearL2Stat::getXTy() const {
    EMx res(maxUpdatedPos_, 1);
    fillXTy(res);
    return res;
}

void LinearL2Stat::fillSumX(EMx& sumX) const {
    for (int i = 0; i < maxUpdatedPos_; ++i) {
        sumX(i, 0) = sumX_[i];
    }
}

LinearL2Stat::EMx LinearL2Stat::getSumX() const {
    EMx res(maxUpdatedPos_, 1);
    fillSumX(res);
    return res;
}

LinearL2Stat::EMx LinearL2Stat::getWHat(double l2reg) const {
    EMx XTX = getXTX();

    if (w_ < 1e-6) {
        auto w = EMx(XTX.rows(), 1);
        for (int i = 0; i < w.rows(); ++i) {
            w(i, 0) = 0;
        }
        return w;
    }

    EMx XTX_r = XTX + DiagMx(XTX.rows(), l2reg);
    return XTX_r.inverse() * getXTy();
}


LinearL2GridStat::LinearL2GridStat(int nBins, int size, int filledSize)
        : nBins_(nBins)
        , size_(size)
        , filledSize_(filledSize) {
    stats_.resize(nBins, LinearL2Stat(size, filledSize));
}

void LinearL2GridStat::reset() {
    for (auto& stat : stats_) {
        stat.reset();
    }
}

void LinearL2GridStat::setFilledSize(int filledSize) {
    filledSize_ = filledSize;
    for (auto& stat : stats_) {
        stat.setFilledSize(filledSize_);
    }
}

LinearL2GridStat& LinearL2GridStat::appendImpl(const LinearL2GridStat& other, const LinearL2GridStatOpParams& opParams) {
    assert(nBins_ == other.nBins_);
    for (int i = 0; i < nBins_; ++i) {
        stats_[i].append(other.stats_[i], opParams);
    }
}

LinearL2GridStat& LinearL2GridStat::removeImpl(const LinearL2GridStat& other, const LinearL2GridStatOpParams& opParams) {
    assert(nBins_ == other.nBins_);
    for (int i = 0; i < nBins_; ++i) {
        stats_[i].remove(other.stats_[i], opParams);
    }
}

LinearL2GridStat& LinearL2GridStat::appendImpl(SampleType x, TargetType y, WeightType weight, const LinearL2GridStatOpParams& opParams) {
    int bin = opParams.bin;
    stats_[bin].append(x, y, weight, opParams);
}

LinearL2GridStat& LinearL2GridStat::removeImpl(SampleType x, TargetType y, WeightType weight, const LinearL2GridStatOpParams& opParams) {
    int bin = opParams.bin;
    stats_[bin].remove(x, y, weight, opParams);
}
