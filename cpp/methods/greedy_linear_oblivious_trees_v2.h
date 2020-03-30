#pragma once

#include <unordered_set>
#include <vector>
#include <memory>

#include "optimizer.h"

#include <models/model.h>
#include <models/bin_optimized_model.h>
#include <targets/linear_l2_stat.h>
#include <targets/linear_l2.h>
#include <data/grid.h>
#include <core/multi_dim_array.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>


class GreedyLinearObliviousTreeLearnerV2;


class LinearObliviousTreeLeafV2;

class GreedyLinearObliviousTreeLearnerV2 final
        : public Optimizer {
public:
    explicit GreedyLinearObliviousTreeLearnerV2(GridPtr grid, int32_t maxDepth = 6,
            int biasCol = -1, double l2reg = 0.0)
            : grid_(std::move(grid))
            , biasCol_(biasCol)
            , maxDepth_(maxDepth)
            , l2reg_(l2reg) {
    }

    GreedyLinearObliviousTreeLearnerV2(const GreedyLinearObliviousTreeLearnerV2& other) = default;

    ModelPtr fit(const DataSet& dataSet, const Target& target) override;

private:
    void cacheDs(const DataSet& ds);

    using TSplit = std::pair<int32_t, int32_t>;

    void buildRoot(const BinarizedDataSet& bds,
                   const DataSet& ds,
                   ConstVecRef<float> ys,
                   ConstVecRef<float> ws);
    void updateNewCorrelations(
            const BinarizedDataSet& bds,
            const DataSet& ds,
            ConstVecRef<float> ys,
            ConstVecRef<float> ws);
    TSplit findBestSplit(const Target& target);
    void initNewLeaves(TSplit split);
    void updateNewLeaves(const BinarizedDataSet& bds,
                         const DataSet& ds,
                         int oldNUsedFeatures,
                         ConstVecRef<float> ys,
                         ConstVecRef<float> ws);

    void resetState();

    template <typename Stat, typename UpdaterT>
    MultiDimArray<2, Stat> ComputeStats(
            int nLeaves, const std::vector<int>& lIds,
            const DataSet& ds, const BinarizedDataSet& bds,
            const Stat& defaultVal,
            UpdaterT updater) {
        int nUsedFeatures = usedFeaturesOrdered_.size();
        std::set<int> usedFeaturesSet(usedFeaturesOrdered_.begin(), usedFeaturesOrdered_.end());

        MultiDimArray<3, Stat> stats({nThreads_, nLeaves, totalBins_}, defaultVal);
        std::vector<std::vector<float>> curX(nThreads_, std::vector<float>(nUsedFeatures, 0.));

        std::cout << 1 << std::endl;

        // compute stats per [thread Id][leaf Id]
        parallelFor(0, nSamples_, [&](int thId, int sampleId) {
            auto& x = curX[thId];
            auto bins = bds.sampleBins(sampleId);
            int lId = lIds[sampleId];
            if (lId < 0) return;
            auto leafStats = stats[thId][lId];

            for (int fId = 0; fId < fCount_; ++fId) {
                int bin = binOffsets_[fId] + bins[fId];
                auto& stat = leafStats[bin];
                updater(stat, x, sampleId, fId);
            }
        });

        std::cout << 2 << std::endl;

        // gather individual workers results together
        parallelFor(0, nLeaves, [&](int lId) {
            for (int thId = 1; thId < nThreads_; ++thId) {
                for (int bin = 0; bin < totalBins_; ++bin) {
//                    std::cout << "+ [" << thId << "][" << lId << "][" << bin << "]" << std::endl;
                    stats[0][lId][bin] += stats[thId][lId][bin];
                }
            }
        });

        std::cout << 3 << std::endl;

        // prefix sum
        parallelFor(0, fCount_, [&](int fId) {
            int offset = binOffsets_[fId];
            const int condCount = grid_->conditionsCount(fId);
            for (int lId = 0; lId < nLeaves; ++lId) {
                auto leafStats = stats[0][lId];
                for (int bin = 1; bin <= condCount; ++bin) {
                    int absBin = offset + bin;
                    leafStats[absBin] += leafStats[absBin - 1];
                }
            }
        });

        std::cout << 4 << std::endl;

        return stats[0].copy();
    }

private:
    GridPtr grid_;
    int32_t maxDepth_ = 6;
    int biasCol_ = -1;
    double l2reg_ = 0.0;

    bool isDsCached_ = false;
    std::vector<Vec> fColumns_;
    std::vector<ConstVecRef<float>> fColumnsRefs_;

    std::vector<int32_t> leafId_;
    std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves_;
    std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> newLeaves_;

    std::set<int> usedFeatures_;
    std::vector<int> usedFeaturesOrdered_;

    std::vector<bool> fullUpdate_;
    std::vector<int> samplesLeavesCnt_;

    ConstVecRef<int32_t> binOffsets_;
    int nThreads_;
    int totalBins_;
    int totalCond_;
    int fCount_;
    int nSamples_;
};
