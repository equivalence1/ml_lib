#pragma once

#include <unordered_set>
#include <vector>
#include <memory>

#include "optimizer.h"
#include <models/model.h>
#include <data/grid.h>
#include <models/bin_optimized_model.h>

class HistogramV2 {
public:
    explicit HistogramV2(BinarizedDataSet& bds, GridPtr grid, unsigned int histSize, unsigned int nUsedFeatures,
            int lastUsedFeatureId);

    void build(const DataSet& ds, const std::vector<int32_t>& indices);
    void updateBin(int64_t fId, int64_t binId, const Vec& x, double y, double f, int offset);
    void prefixSumBins();
    void prefixSumBinsLastFeature(int corOffset);

    std::pair<double, double> splitScore(int fId, int condId, double l2reg, double traceReg);

    std::shared_ptr<Mx> getW(double l2reg);

    void printEig(double l2reg);
    void printCnt();
    void print();

private:
    static double computeScore(Mx& XTX, Mx& XTy, double XTX_trace, uint32_t cnt, double l2reg,
                               double traceReg);

    static void printEig(Mx& M);

    friend HistogramV2 operator-(const HistogramV2& lhs, const HistogramV2& rhs);

private:
    GridPtr grid_;

    std::vector<Mx> hist_XTX_; // (X^T * X) for bins
    std::vector<Mx> hist_XTy_; // (X^T * y) for bins
    std::vector<double> hist_XTX_trace_;
    std::vector<uint32_t> hist_cnt_;

    int lastUsedFeatureId_ = -1;
    unsigned int nUsedFeatures_;
    unsigned int histSize_;

    BinarizedDataSet& bds_;
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
    GridPtr grid_;
    int32_t maxDepth_ = 6;
    int biasCol_ = -1;
    double l2reg_ = 0.0;
    double traceReg_ = 0.0;
};

class LinearObliviousTreeV2 final
        : public Stub<Model, LinearObliviousTreeV2>
        , std::enable_shared_from_this<LinearObliviousTreeV2> {
public:

    LinearObliviousTreeV2(const LinearObliviousTreeV2& other, double scale)
            : Stub<Model, LinearObliviousTreeV2>(other.gridPtr()->origFeaturesCount(), 1) {
        grid_ = other.grid_;
        usedFeatures_ = other.usedFeatures_;
        scale_ = scale;
        leaves_ = other.leaves_;
    }

    LinearObliviousTreeV2(GridPtr grid, std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves)
            : Stub<Model, LinearObliviousTreeV2>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , leaves_(std::move(leaves)) {
        scale_ = 1;
    }

    LinearObliviousTreeV2(GridPtr grid)
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
    std::set<int64_t> usedFeatures_;
    double scale_ = 1;

    std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves_;
};
