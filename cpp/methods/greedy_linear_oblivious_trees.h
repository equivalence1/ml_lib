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
    explicit Histogram(GridPtr grid) : grid_(std::move(grid)) {

    }

    void build(const DataSet& ds, const std::set<int>& usedFeatures,
            const std::vector<int32_t>& indices);

    std::pair<double, double> splitScore(int fId, int condId);

    std::shared_ptr<Mx> getW();

private:
    static double computeScore(Mx& XTX, Mx& XTy, uint32_t cnt);

    friend Histogram operator-(const Histogram& lhs, const Histogram& rhs);

private:
    GridPtr grid_;

    // (X^T * X) for bins
    std::vector<Mx> hist_XTX_;
    std::vector<Mx> histLeft_XTX_;

    // (X^T * y) for bins
    std::vector<Mx> hist_XTy_;
    std::vector<Mx> histLeft_XTy_;

    std::vector<uint32_t> hist_cnt_;
    std::vector<uint32_t> histLeft_cnt_;

    int usedFeature_ = -1; // any of them...
};

class LinearObliviousTreeLeaf;

class GreedyLinearObliviousTreeLearner final
        : public Optimizer {
public:
    explicit GreedyLinearObliviousTreeLearner(GridPtr grid, int32_t maxDepth = 6)
            : grid_(std::move(grid))
            , maxDepth_(maxDepth) {
    }

    GreedyLinearObliviousTreeLearner(const GreedyLinearObliviousTreeLearner& other) = default;

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

private:
    GridPtr grid_;
    int32_t maxDepth_ = 6;
};

class GreedyLinearObliviousTree final
        : public Stub<Model, GreedyLinearObliviousTree>
        , std::enable_shared_from_this<GreedyLinearObliviousTree> {
public:

    GreedyLinearObliviousTree(const GreedyLinearObliviousTree& other, double scale)
            : Stub<Model, GreedyLinearObliviousTree>(other.gridPtr()->origFeaturesCount(), 1) {
        grid_ = other.grid_;
        usedFeatures_ = other.usedFeatures_;
        scale_ = scale;
        leaves_ = other.leaves_;
    }

    GreedyLinearObliviousTree(
            GridPtr grid,
            std::vector<std::shared_ptr<LinearObliviousTreeLeaf>> leaves)
            : Stub<Model, GreedyLinearObliviousTree>(grid->origFeaturesCount(), 1)
            , grid_(std::move(grid))
            , leaves_(std::move(leaves)) {
        scale_ = 1;
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
    double value(const Vec& x) const;

    friend class GreedyLinearObliviousTreeLearner;

private:
    GridPtr grid_;
    std::set<int> usedFeatures_;
    double scale_ = 1;

    std::vector<std::shared_ptr<LinearObliviousTreeLeaf>> leaves_;
};
