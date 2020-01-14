#pragma once

#include <unordered_set>
#include <vector>
#include <memory>

#include "optimizer.h"
#include <models/model.h>
#include <data/grid.h>
#include <models/bin_optimized_model.h>

class Histogram {
public:
    explicit Histogram(BinarizedDataSetPtr& bds) : bds_(bds) {

    }

    void build(const DataSet& ds, const std::set<int>& usedFeatures,
            const std::vector<int32_t>& indices);

    std::pair<double, double> splitScore(int fId, int condId);

private:
    static double computeScore(Mx& XTX, Mx& XTy, uint32_t cnt);

    friend Histogram operator-(const Histogram& lhs, const Histogram& rhs);

private:
    BinarizedDataSetPtr& bds_;

    // (X^T * X) for bins
    std::vector<Mx> hist_XTX_;
    std::vector<Mx> histLeft_XTX_;

    // (X^T * y) for bins
    std::vector<Mx> hist_XTy_;
    std::vector<Mx> histLeft_XTy_;

    std::vector<uint32_t> hist_cnt_;
    std::vector<uint32_t> histLeft_cnt_;
};

class LinearObliviousTreeLeaf;

class GreedyLinearObliviousTree : public Optimizer
//                                , public Stub<BinOptimizedModel, GreedyLinearObliviousTree>
                                , std::enable_shared_from_this<GreedyLinearObliviousTree>  {
public:
    explicit GreedyLinearObliviousTree(BinarizedDataSetPtr& bds, int32_t maxDepth = 6)
            : //Stub<BinOptimizedModel, GreedyLinearObliviousTree>(bds->grid().origFeaturesCount(), 1)
             bds_(bds)
            , maxDepth_(maxDepth) {

    }

//    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

    Grid grid() const {
        return bds_->grid();
    }

    GridPtr gridPtr() const {
        return bds_->gridPtr();
    }

//    void appendTo(const Vec& x, Vec to) const override;
//
//    void applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const override;
//
//    void applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const;
//
//    double value(const Vec& x) override;
//
//    void grad(const Vec& x, Vec to) override;

protected:
    friend class LinearObliviousTreeLeaf;

    BinarizedDataSetPtr& bds_;

private:
    int32_t maxDepth_ = 6;
    std::set<int> usedFeatures_;
    int32_t curDepth_ = 0;

    std::vector<std::shared_ptr<LinearObliviousTreeLeaf>> leaves_;
};