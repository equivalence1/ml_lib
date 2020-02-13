#pragma once

#include <unordered_set>
#include <vector>
#include <memory>

#include "optimizer.h"
#include <models/model.h>
#include <data/grid.h>
#include <models/bin_optimized_model.h>

class GreedyLinearObliviousTreeLearnerV2;


class BinStat {
public:
    explicit BinStat(int size, int filledSize)
            : size_(size)
            , filledSize_(filledSize) {
        for (int i = 0; i < size; ++i) {
            XTX_.emplace_back(i + 1, 0.0);
        }
        XTy_ = std::vector<double>(size, 0.0);
        cnt_ = 0;
        trace_ = 0.0;
        maxUpdatedPos_ = filledSize_;
    }

    void reset() {
        cnt_ = 0;
        trace_ = 0;
        for (int i = 0; i <= filledSize_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX_[i][j] = 0;
            }
            XTy_[i] = 0;
        }
        filledSize_ = 0;
    }

    void setFilledSize(int filledSize) {
        filledSize_ = filledSize;
    }

    int filledSize() {
        return filledSize_;
    }

    void addNewCorrelation(const std::vector<double>& xtx, double xty) {
        assert(xtx.size() >= filledSize_ + 1);

        for (int i = 0; i <= filledSize_; ++i) {
            XTX_[filledSize_][i] += xtx[i];
        }
        XTy_[filledSize_] += xty;
        trace_ += xtx[filledSize_];
        maxUpdatedPos_ = filledSize_ + 1;
    }

    void addFullCorrelation(Vec x, double y) {
        assert(x.size() >= filledSize_);

        auto xRef = x.arrayRef();

        for (int i = 0; i < filledSize_; ++i) {
            XTy_[i] += xRef[i] * y;
        }

        for (int i = 0; i < filledSize_; ++i) {
            for (int j = i; j < filledSize_; ++j) {
                XTX_[j][i] += xRef[i] * xRef[j]; // TODO change order, this one is bad for caches
            }
        }

        cnt_ += 1;
    }

    Mx getXTX() const {
        Mx res(maxUpdatedPos_, maxUpdatedPos_);
        auto resRef = res.arrayRef();

        for (int i = 0; i < maxUpdatedPos_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                int pos = i * maxUpdatedPos_ + j;
                resRef[pos] = XTX_[i][j];
                pos = j * maxUpdatedPos_ + i;
                resRef[pos] = XTX_[i][j];
            }
        }

        return res;
    }

    Mx getXTy() const {
        Mx res(maxUpdatedPos_, 1);
        auto resRef = res.arrayRef();

        for (int i = 0; i < maxUpdatedPos_; ++i) {
            resRef[i] = XTy_[i];
        }

        return res;
    }

    uint32_t getCnt() {
        return cnt_;
    }

    double getTrace() {
        return trace_;
    }

    // This one DOES NOT add up new correlations
    BinStat& operator+=(const BinStat& s) {
        assert(filledSize_ == s.filledSize_);

        cnt_ += s.cnt_;
        trace_ += s.trace_;

        for (int i = 0; i < filledSize_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX_[i][j] += s.XTX_[i][j];
            }
            XTy_[i] += s.XTy_[i];
        }
    }

    // This one DOES NOT subtract new correlations
    BinStat& operator-=(const BinStat& s) {
        cnt_ -= s.cnt_;
        trace_ -= s.trace_;

        for (int i = 0; i < filledSize_; ++i) {
            for (int j = 0; j < i + 1; ++j) {
                XTX_[i][j] -= s.XTX_[i][j];
            }
            XTy_[i] -= s.XTy_[i];
        }
    }

private:
    friend BinStat operator+(const BinStat& lhs, const BinStat& rhs);
    friend BinStat operator-(const BinStat& lhs, const BinStat& rhs);

private:
    int size_;
    int filledSize_;
    int maxUpdatedPos_;

    std::vector<std::vector<double>> XTX_;
    std::vector<double> XTy_;
    uint32_t cnt_;
    double trace_;
};

inline BinStat operator+(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res += rhs;
    return res;
}

inline BinStat operator-(const BinStat& lhs, const BinStat& rhs) {
    BinStat res(lhs);
    res -= rhs;
    return res;
}





class HistogramV2 {
public:
    HistogramV2(BinarizedDataSet& bds, GridPtr grid, unsigned int nUsedFeatures, int lastUsedFeatureId);

    void addFullCorrelation(int bin, Vec x, double y);
    void addNewCorrelation(int bin, const std::vector<double>& xtx, double xty);
    void prefixSumBins();

    void addBinStat(int bin, const BinStat& stats);

    std::pair<double, double> splitScore(int fId, int condId, double l2reg, double traceReg);

    std::shared_ptr<Mx> getW(double l2reg);

    void printEig(double l2reg);
    void printCnt();
    void print();

    HistogramV2& operator+=(const HistogramV2& h);
    HistogramV2& operator-=(const HistogramV2& h);

private:
    static double computeScore(Mx& XTX, Mx& XTy, double XTX_trace, uint32_t cnt, double l2reg,
                               double traceReg);

    static void printEig(Mx& M);

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

private:
    GridPtr grid_;
    int32_t maxDepth_ = 6;
    int biasCol_ = -1;
    double l2reg_ = 0.0;
    double traceReg_ = 0.0;

    bool isDsCached_ = false;
    std::vector<Vec> fColumns_;
    std::vector<ConstVecRef<float>> fColumnsRefs_;

    // thread      leaf         bin         coordinate
    std::vector<std::vector<std::vector<std::vector<double>>>> h_XTX_;
    std::vector<std::vector<std::vector<double>>> h_XTy_;
    std::vector<std::vector<std::vector<BinStat>>> stats_;

    ConstVecRef<int32_t> binOffsets_;
    int nThreads_;
    int totalBins_;
    int fCount_;
};

class LinearObliviousTreeV2 final
        : public Stub<Model, LinearObliviousTreeV2>
        , std::enable_shared_from_this<LinearObliviousTreeV2> {
public:

    LinearObliviousTreeV2(const LinearObliviousTreeV2& other, double scale)
            : Stub<Model, LinearObliviousTreeV2>(other.gridPtr()->origFeaturesCount(), 1) {
        grid_ = other.grid_;
        scale_ = scale;
        leaves_ = other.leaves_;
    }

    LinearObliviousTreeV2(GridPtr grid, std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves)
            : Stub<Model, LinearObliviousTreeV2>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , leaves_(std::move(leaves)) {
        scale_ = 1;
    }

    explicit LinearObliviousTreeV2(GridPtr grid)
            : Stub<Model, LinearObliviousTreeV2>(grid->origFeaturesCount(), 1)
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

//    void applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const override;

//    void applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const;

    double value(const Vec& x) override;

    void grad(const Vec& x, Vec to) override;

private:
    friend class GreedyLinearObliviousTreeLearnerV2;

    double value(const Vec& x) const;

private:
    GridPtr grid_;
    double scale_ = 1;

    std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves_;
};
