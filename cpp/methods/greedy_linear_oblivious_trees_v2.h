#pragma once

#include <unordered_set>
#include <vector>
#include <memory>

#include "optimizer.h"

#include <models/model.h>
#include <models/bin_optimized_model.h>

#include <data/grid.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>


class GreedyLinearObliviousTreeLearnerV2;

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

class BinStat {
public:
    using EMx = Eigen::MatrixXd;
    using fType = float;

    explicit BinStat(int size, int filledSize)
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

    void reset() {
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

    void setFilledSize(int filledSize) {
        filledSize_ = filledSize;
        maxUpdatedPos_ = filledSize_;
    }

    [[nodiscard]] int filledSize() const {
        return filledSize_;
    }

    [[nodiscard]] int mxSize() const {
        return maxUpdatedPos_;
    }

    void addNewCorrelation(const std::vector<float>& xtx, float xty, int shift = 0) {
        const int corPos = filledSize_ + shift;

        int pos = corPos * (corPos + 1) / 2;
        for (int i = 0; i <= corPos; ++i) {
            xtx_[pos + i] += xtx[i];
        }
        xty_[corPos] += xty;
        maxUpdatedPos_ = std::max(maxUpdatedPos_, corPos + 1);
    }

    void addFullCorrelation(const std::vector<float>& x, float y, float w) {
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

        w_ += w;
        float yw = y * w;
        sumY_ += yw;
        sumY2_ += yw * y;
    }

    void fillXTX(EMx& XTX) const {
        int basePos = 0;
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX(i, j) = xtx_[basePos + j];
                XTX(j, i) = xtx_[basePos + j];
            }
            basePos += i + 1;
        }
    }

    [[nodiscard]] EMx getXTX() const {
        EMx res(maxUpdatedPos_, maxUpdatedPos_);
        fillXTX(res);
        return res;
    }

    void fillXTy(EMx& XTy) const {
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            XTy(i, 0) = xty_[i];
        }
    }

    [[nodiscard]] EMx getXTy() const {
        EMx res(maxUpdatedPos_, 1);
        fillXTy(res);
        return res;
    }

    void fillSumX(EMx& sumX) const {
        for (int i = 0; i < maxUpdatedPos_; ++i) {
            sumX(i, 0) = sumX_[i];
        }
    }

    [[nodiscard]] EMx getSumX() const {
        EMx res(maxUpdatedPos_, 1);
        fillSumX(res);
        return res;
    }

    [[nodiscard]] EMx getW(double l2reg) const {
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

    [[nodiscard]] float getWeight() const {
        return w_;
    }

    double fitScore(double l2reg, bool log = false) {
//        if (log) std::cout << "fitScoring, " << maxUpdatedPos_ << ", l2=" << l2reg << std::endl;
//        if (log) std::cout << "getXTX():\n" << getXTX() << "\nDiagMx:\n" << DiagMx(maxUpdatedPos_, l2reg) << "\n+\n"<< getXTX() + DiagMx(maxUpdatedPos_, l2reg) << std::endl;
//        Eigen::MatrixXd XTX = getXTX() + DiagMx(maxUpdatedPos_, l2reg);
//        if (log) std::cout << "XTX=\n" << XTX << std::endl;
//        Eigen::MatrixXd XTy = getXTy();
//        if (log) std::cout << "XTy=\n" << XTy << std::endl;
//
//        Eigen::MatrixXd w = XTX.inverse() * XTy;
//        if (log) std::cout << "w=\n" << w << std::endl;
//
//        Eigen::MatrixXd c1(XTy.transpose() * w);
//        c1 *= -2;
//        assert(c1.rows() == 1 && c1.cols() == 1);
//        if (log) std::cout << "c1=\n" << c1 << std::endl;
//
//        Eigen::MatrixXd c2(w.transpose() * XTX * w);
//        assert(c2.rows() == 1 && c2.cols() == 1);
//        if (log) std::cout << "c2=\n" << c2 << std::endl;
//
//        Eigen::MatrixXd reg = w.transpose() * w * l2reg;
//        assert(reg.rows() == 1 && reg.cols() == 1);
//        if (log) std::cout << "reg=\n" << reg << std::endl;
//
//        Eigen::MatrixXd res = c1 + c2 + reg;
//        if (log) std::cout << "res=\n" << reg << std::endl;
//
//        return res(0, 0);

        if (w_ < 2) {
            return 0;
        }

        EMx wHat = getW(l2reg);

        EMx xty = getXTy();
        xty -= (sumY_ / w_) * getSumX();
//        xty *= 1.0 / w;

        float reg = 1 + 0.005f * std::log(w_ + 1);

        float scoreFromLinear = (xty.transpose() * wHat)(0, 0);
        float scoreFromConst = (sumY_ * sumY_) / w_;
        float targetValue = scoreFromConst + scoreFromLinear - l2reg * (wHat.transpose() * wHat)(0, 0);

        return -targetValue * reg;
    }

    BinStat& addSized(const BinStat& s, int size) {
        w_ += s.w_;
        sumY_ += s.sumY_;
        sumY2_ += s.sumY2_;
        trace_ += s.trace_;

        int pos = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                xtx_[pos + j] += s.xtx_[pos + j];
            }
            pos += i + 1;
            xty_[i] += s.xty_[i];
            sumX_[i] += s.sumX_[i];
        }

        return *this;
    }

    BinStat& subtractSized(const BinStat& s, int size) {
        w_ -= s.w_;
        sumY_ -= s.sumY_;
        sumY2_ -= s.sumY2_;
        trace_ -= s.trace_;

        int pos = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                xtx_[pos + j] -= s.xtx_[pos + j];
            }
            pos += i + 1;
            xty_[i] -= s.xty_[i];
            sumX_[i] -= s.sumX_[i];
        }

        return *this;
    }

    BinStat& operator+=(const BinStat& s) {
        return addSized(s, std::min(filledSize_, s.filledSize_));
    }

    BinStat& operator-=(const BinStat& s) {
        return subtractSized(s, std::min(filledSize_, s.filledSize_));
    }

private:
//    friend BinStat operator+(const BinStat& lhs, const BinStat& rhs);
//    friend BinStat operator-(const BinStat& lhs, const BinStat& rhs);

    friend BinStat subtractFull(const BinStat& lhs, const BinStat& rhs);
    friend BinStat addFull(const BinStat& lhs, const BinStat& rhs);

    friend BinStat subtractFilled(const BinStat& lhs, const BinStat& rhs);
    friend BinStat addFilled(const BinStat& lhs, const BinStat& rhs);

public:
    int size_;
    int filledSize_;
    int maxUpdatedPos_;

    float w_;
    float trace_;
    float sumY_;
    float sumY2_;
    std::vector<float> xtx_;
    std::vector<float> xty_;
    std::vector<float> sumX_;
};

inline BinStat addFilled(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res += rhs;
    return res;
}

inline BinStat subtractFilled(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res -= rhs;
    return res;
}

inline BinStat subtractFull(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res.maxUpdatedPos_ = std::max(lhs.maxUpdatedPos_, rhs.maxUpdatedPos_);
    return res.subtractSized(rhs, rhs.maxUpdatedPos_);
}

inline BinStat addFull(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res.maxUpdatedPos_ = std::max(lhs.maxUpdatedPos_, rhs.maxUpdatedPos_);
    return res.subtractSized(rhs, rhs.maxUpdatedPos_);
}



class HistogramV2 {
public:
    HistogramV2(BinarizedDataSet& bds, GridPtr grid, unsigned int nUsedFeatures, int lastUsedFeatureId);

//    void addFullCorrelation(int bin, Vec x, double y);
    void addNewCorrelation(int bin, const std::vector<float>& xtx, float xty, int shift = 0);
    void prefixSumBins();

    void addBinStat(int bin, const BinStat& stats);

    std::pair<double, double> splitScore(int fId, int condId, double l2reg, double traceReg);

    std::shared_ptr<Eigen::MatrixXd> getW(double l2reg);

    void printEig(double l2reg);
    void printCnt();
    void print();

    HistogramV2& operator+=(const HistogramV2& h);
    HistogramV2& operator-=(const HistogramV2& h);

private:
    static double computeScore(Eigen::MatrixXd XTX, Eigen::MatrixXd XTy, double l2reg, bool log = false);

    static void printEig(Eigen::MatrixXd& M);

    friend HistogramV2 operator-(const HistogramV2& lhs, const HistogramV2& rhs);
    friend HistogramV2 operator+(const HistogramV2& lhs, const HistogramV2& rhs);

private:
    BinarizedDataSet& bds_;
    GridPtr grid_;

    std::vector<BinStat> hist_;

    int lastUsedFeatureId_ = -1;
    unsigned int nUsedFeatures_;

    friend class GreedyLinearObliviousTreeLearnerV2;
};

class LinearObliviousTreeLeafV2;

class GreedyLinearObliviousTreeLearnerV2 final
        : public Optimizer {
public:
    explicit GreedyLinearObliviousTreeLearnerV2(GridPtr grid, int32_t maxDepth = 6, int biasCol = -1,
                                                double l2reg = 0.0, double traceReg = 0.0)
            : grid_(std::move(grid))
            , biasCol_(biasCol)
            , maxDepth_(maxDepth)
            , l2reg_(l2reg)
            , traceReg_(traceReg) {
    }

    GreedyLinearObliviousTreeLearnerV2(const GreedyLinearObliviousTreeLearnerV2& other) = default;

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

private:
    void cacheDs(const DataSet& ds);
    void resetState();

private:
    GridPtr grid_;
    int32_t maxDepth_ = 6;
    int biasCol_ = -1;
    double l2reg_ = 0.0;
    double traceReg_ = 0.0;

    bool isDsCached_ = false;
    std::vector<Vec> fColumns_;
    std::vector<ConstVecRef<float>> fColumnsRefs_;
    std::vector<std::vector<float>> curX_;

    std::vector<int32_t> leafId_;

    std::set<int> usedFeatures_;
    std::vector<int> usedFeaturesOrdered_;

    // thread      leaf         bin         coordinate
    std::vector<std::vector<std::vector<std::vector<float>>>> h_XTX_;
    std::vector<std::vector<std::vector<float>>> h_XTy_;
    std::vector<std::vector<std::vector<BinStat>>> stats_;

    std::vector<bool> fullUpdate_;
    std::vector<int> samplesLeavesCnt_;

    ConstVecRef<int32_t> binOffsets_;
    int nThreads_;
    int totalBins_;
    int totalCond_;
    int fCount_;
    int nSamples_;
};

class LinearObliviousTreeV2 final
        : public Stub<BinOptimizedModel, LinearObliviousTreeV2>
                , std::enable_shared_from_this<LinearObliviousTreeV2> {
public:

    LinearObliviousTreeV2(const LinearObliviousTreeV2& other, double scale)
            : Stub<BinOptimizedModel, LinearObliviousTreeV2>(other.gridPtr()->origFeaturesCount(), 1) {
        grid_ = other.grid_;
        scale_ = scale;
        leaves_ = other.leaves_;
        splits_ = other.splits_;
    }

    LinearObliviousTreeV2(GridPtr grid, std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves)
            : Stub<BinOptimizedModel, LinearObliviousTreeV2>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , leaves_(std::move(leaves)) {
        scale_ = 1;
    }

    explicit LinearObliviousTreeV2(GridPtr grid)
            : Stub<BinOptimizedModel, LinearObliviousTreeV2>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid)) {

    }

    Grid grid() const {
        return *grid_.get();
    }

    GridPtr gridPtr() const {
        return grid_;
    }

    // I have now idea what this function should do...
    // For now just adding value(x) to @param to.
    void appendTo(const Vec& x, Vec to) const override;

    void applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const override;

    void applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const {
        throw std::runtime_error("Unimplemented");
    }

    double value(const Vec& x) override;

    void grad(const Vec& x, Vec to) override;

private:
    friend class GreedyLinearObliviousTreeLearnerV2;

    double value(const ConstVecRef<float>& x) const;

private:
    GridPtr grid_;
    double scale_ = 1;

    std::vector<std::tuple<int32_t, int32_t>> splits_;

    std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves_;
};
