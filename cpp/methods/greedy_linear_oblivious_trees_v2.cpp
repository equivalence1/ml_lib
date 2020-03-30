#include "greedy_linear_oblivious_trees_v2.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <chrono>

#include <core/vec_factory.h>
#include <core/matrix.h>
#include <core/multi_dim_array.h>

#include <models/linear_oblivious_tree.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>


#define TIME_BLOCK_START(name) \
    auto begin##name = std::chrono::steady_clock::now();

#define TIME_BLOCK_END(name) \
    do { \
        auto end##name = std::chrono::steady_clock::now(); \
        auto time_ms##name = std::chrono::duration_cast<std::chrono::milliseconds>(end##name - begin##name).count(); \
        std::cout << #name << " done in " << time_ms##name << " [ms]" << std::endl; \
    } while (false);


class LinearObliviousTreeLeafV2 : std::enable_shared_from_this<LinearObliviousTreeLeafV2> {
public:
    LinearObliviousTreeLeafV2(
            GridPtr grid,
            int nUsedFeatures)
            : grid_(std::move(grid))
            , nUsedFeatures_(nUsedFeatures)
            , stats_(MultiDimArray<1, LinearL2Stat>({(int)grid_->totalBins()}, nUsedFeatures + 1, nUsedFeatures)) {
//        for (int bin = 0; bin < )
        id_ = 0;
    }

    double splitScore(const StatBasedLoss<LinearL2Stat>& target, int fId, int condId) {
        int bin = grid_->binOffsets()[fId] + condId;
        int lastBin = (int)grid_->binOffsets()[fId] + (int)grid_->conditionsCount(fId);

        auto leftStat = stats_[bin];
        auto rightStat = stats_[lastBin] - leftStat;

        return target.score(leftStat) + target.score(rightStat);
    }

    void fit(float l2reg) {
        w_ = stats_[grid_->totalBins() - 1].getWHat(l2reg);
    }

    double value(const ConstVecRef<float>& x) const {
        float res = 0.0;

        int i = 0;
        for (auto f : usedFeaturesInOrder_) {
            res += x[f] * (float)w_(i, 0);
            ++i;
        }

        return res;
    }

    std::pair<std::shared_ptr<LinearObliviousTreeLeafV2>, std::shared_ptr<LinearObliviousTreeLeafV2>>
    split(int32_t fId, int32_t condId) {
        int origFId = grid_->origFeatureIndex(fId);
        unsigned int nUsedFeatures = nUsedFeatures_ + (1 - usedFeatures_.count(origFId));
//        std::cout << "new nUsedFeatures: " << nUsedFeatures << std::endl;

        auto left = std::make_shared<LinearObliviousTreeLeafV2>(grid_, nUsedFeatures);
        auto right = std::make_shared<LinearObliviousTreeLeafV2>(grid_, nUsedFeatures);

        initChildren(left, right, fId, condId);

        return std::make_pair(left, right);
    }

    void prefixSumBins() {
        for (int fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
            int offset = grid_->binOffsets()[fId];
            for (int bin = 1; bin <= grid_->conditionsCount(fId); ++bin) {
                int absBin = offset + bin;
                stats_[absBin] += stats_[absBin - 1];
            }
        }
    }

//    void printHists() {
//        hist_->print();
//    }
//
//    void printInfo() {
//        hist_->printEig(l2reg_);
//        hist_->printCnt();
//        printSplits();
//        std::cout << std::endl;
//    }

    void printSplits() {
        for (auto& s : splits_) {
            auto fId = std::get<0>(s);
            auto origFId = grid_->origFeatureIndex(fId);
            auto condId = std::get<1>(s);
            double minCondition = grid_->condition(fId, 0);
            double maxCondition = grid_->condition(fId, grid_->conditionsCount(fId) - 1);
            double condition = grid_->condition(fId, condId);
            std::cout << "split: fId=" << fId << "(" << origFId << ") " << ", condId=" << condId
                      << std::setprecision(5) << ", used cond=" << condition
                      << ", min cond=" << minCondition << ", max cond=" << maxCondition << std::endl;
        }
    }

private:
    void initChildren(std::shared_ptr<LinearObliviousTreeLeafV2>& left,
                      std::shared_ptr<LinearObliviousTreeLeafV2>& right,
                      int32_t splitFId, int32_t condId) {
        left->id_ = 2 * id_;
        right->id_ = 2 * id_ + 1;

        left->usedFeatures_ = usedFeatures_;
        right->usedFeatures_ = usedFeatures_;
        left->usedFeaturesInOrder_ = usedFeaturesInOrder_;
        right->usedFeaturesInOrder_ = usedFeaturesInOrder_;

        int32_t origFeatureId = grid_->origFeatureIndex(splitFId);

        if (usedFeatures_.count(origFeatureId) == 0) {
            left->usedFeatures_.insert(origFeatureId);
            right->usedFeatures_.insert(origFeatureId);
            left->usedFeaturesInOrder_.push_back(origFeatureId);
            right->usedFeaturesInOrder_.push_back(origFeatureId);
        }

        left->splits_ = this->splits_;
        left->splits_.emplace_back(std::make_tuple(splitFId, condId, true));
        right->splits_ = this->splits_;
        right->splits_.emplace_back(std::make_tuple(splitFId, condId, false));
    }

private:
    friend class GreedyLinearObliviousTreeLearnerV2;

    GridPtr grid_;
    std::set<int32_t> usedFeatures_;
    std::vector<int32_t> usedFeaturesInOrder_;
    LinearL2Stat::EMx w_;
    std::vector<std::tuple<int32_t, int32_t, bool>> splits_;
    MultiDimArray<1, LinearL2Stat> stats_;

    unsigned int nUsedFeatures_;

    int32_t id_;
};



ModelPtr GreedyLinearObliviousTreeLearnerV2::fit(const DataSet& ds, const Target& target) {
    auto beginAll = std::chrono::steady_clock::now();

    auto tree = std::make_shared<LinearObliviousTree>(grid_);

    cacheDs(ds);
    resetState();

    // todo cache
    auto bds = cachedBinarize(ds, grid_, fCount_);

    auto ysVec = target.targets();
    auto ys = ysVec.arrayRef();

    auto wsVec = target.weights();
    auto ws = wsVec.arrayRef();

    if (biasCol_ == -1) {
        // TODO
        throw std::runtime_error("provide bias col!");
    }

    std::cout << "start" << std::endl;


    TIME_BLOCK_START(BUILDING_ROOT)
    buildRoot(bds, ds, ys, ws);
    TIME_BLOCK_END(BUILDING_ROOT)

    // Root is built

    for (unsigned int d = 0; d < maxDepth_; ++d) {
        TIME_BLOCK_START(UPDATE_NEW_CORRELATIONS)
        updateNewCorrelations(bds, ds, ws, ys);
        TIME_BLOCK_END(UPDATE_NEW_CORRELATIONS)

        TIME_BLOCK_START(FIND_BEST_SPLIT)
        auto split = findBestSplit(target);
        int32_t splitFId = split.first;
        int32_t splitCond = split.second;
        tree->splits_.emplace_back(std::make_tuple(splitFId, splitCond));

        int oldNUsedFeatures = usedFeatures_.size();

        int32_t splitOrigFId = grid_->origFeatureIndex(splitFId);
        if (usedFeatures_.count(splitOrigFId) == 0) {
            usedFeatures_.insert(splitOrigFId);
            usedFeaturesOrdered_.push_back(splitOrigFId);
        }
        TIME_BLOCK_END(FIND_BEST_SPLIT)

        TIME_BLOCK_START(INIT_NEW_LEAVES)
        initNewLeaves(split);
        TIME_BLOCK_END(INIT_NEW_LEAVES)

        TIME_BLOCK_START(UPDATE_NEW_LEAVES)
        updateNewLeaves(bds, ds, oldNUsedFeatures, ys, ws);
        TIME_BLOCK_END(UPDATE_NEW_LEAVES)

        leaves_ = std::move(newLeaves_);
    }

    TIME_BLOCK_START(FINAL_FIT)
    parallelFor(0, leaves_.size(), [&](int lId) {
        auto& l = leaves_[lId];
        l->fit((float)l2reg_);
    });
    TIME_BLOCK_END(FINAL_FIT)

    std::vector<LinearObliviousTreeLeaf> inferenceLeaves;
    for (auto& l : leaves_) {
        inferenceLeaves.emplace_back(usedFeaturesOrdered_, l->w_);
    }

    tree->leaves_ = std::move(inferenceLeaves);
    return tree;
}

void GreedyLinearObliviousTreeLearnerV2::cacheDs(const DataSet &ds) {

    // TODO this "caching" prevents from using bootstrap and rsm, but at least makes default boosting faster for now...

    if (isDsCached_) {
        return;
    }

    for (int fId = 0; fId < (int)grid_->nzFeaturesCount(); ++fId) {
        fColumns_.emplace_back(ds.samplesCount());
        fColumnsRefs_.emplace_back(NULL);
    }

    parallelFor<0>(0, grid_->nzFeaturesCount(), [&](int fId) {
        int origFId = grid_->origFeatureIndex(fId);
        ds.copyColumn(origFId, &fColumns_[fId]);
        fColumnsRefs_[fId] = fColumns_[fId].arrayRef();
    });

    totalBins_ = grid_->totalBins();
    fCount_ = grid_->nzFeaturesCount();
    totalCond_ = totalBins_ - fCount_;
    binOffsets_ = grid_->binOffsets();
    nThreads_ = (int)GlobalThreadPool<0>().numThreads();
    nSamples_ = ds.samplesCount();

    fullUpdate_.resize(1U << (unsigned)maxDepth_, false);
    samplesLeavesCnt_.resize(1U << (unsigned)maxDepth_, 0);

    isDsCached_ = true;
}

void GreedyLinearObliviousTreeLearnerV2::resetState() {
    usedFeatures_.clear();
    usedFeaturesOrdered_.clear();
    leafId_.clear();
    leafId_.resize(nSamples_, 0);
}

void GreedyLinearObliviousTreeLearnerV2::buildRoot(
        const BinarizedDataSet &bds,
        const DataSet &ds,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    auto root = std::make_shared<LinearObliviousTreeLeafV2>(this->grid_, 1);
    root->usedFeatures_.insert(biasCol_);
    root->usedFeaturesInOrder_.push_back(biasCol_);

    usedFeatures_.insert(biasCol_);
    usedFeaturesOrdered_.push_back(biasCol_);

    LinearL2StatOpParams params;
    params.vecAddMode = LinearL2StatOpParams::FullCorrelation;

    MultiDimArray<2, LinearL2Stat> stats = ComputeStats<LinearL2Stat>(
            1, leafId_, ds, bds,
            LinearL2Stat(2, 1),
            [&](LinearL2Stat& stat, std::vector<float>& x, int sampleId, int fId) {
        ds.fillSample(sampleId, usedFeaturesOrdered_, x);
        stat.append(x.data(), ys[sampleId], ws[sampleId], params);
    });

    root->stats_ = stats[0].copy();

    leaves_.emplace_back(std::move(root));
}

void GreedyLinearObliviousTreeLearnerV2::updateNewCorrelations(
        const BinarizedDataSet& bds,
        const DataSet& ds,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    int nUsedFeatures = usedFeaturesOrdered_.size();
    std::set<int> usedFeaturesSet(usedFeaturesOrdered_.begin(), usedFeaturesOrdered_.end());

    std::cout << "updateNewCorrelations: 1" << std::endl;

    MultiDimArray<2, LinearL2CorStat> stats = ComputeStats<LinearL2CorStat>(
            leaves_.size(), leafId_, ds, bds,
            LinearL2CorStat(nUsedFeatures + 1),
            [&](LinearL2CorStat& stat, std::vector<float>& x, int sampleId, int fId) {
        int origFId = grid_->origFeatureIndex(fId);
        if (usedFeaturesSet.count(origFId)) return;

        ds.fillSample(sampleId, usedFeaturesOrdered_, x);
        LinearL2CorStatOpParams params;
        params.fVal = ds.fVal(sampleId, origFId);

        stat.append(x.data(), ys[sampleId], ws[sampleId], params);
    });

    std::cout << "updateNewCorrelations: 2" << std::endl;

    // update stats with this correlations
    parallelFor(0, totalBins_, [&](int bin) {
        LinearL2StatOpParams params = {};
        for (int lId = 0; lId < leaves_.size(); ++lId) {
            leaves_[lId]->stats_[bin].append(stats[lId][bin].xxt.data(),
                                             stats[lId][bin].xy, /*unused*/1.0, params);
        }
    }, false);
}

GreedyLinearObliviousTreeLearnerV2::TSplit GreedyLinearObliviousTreeLearnerV2::findBestSplit(
        const Target& target) {
    float bestScore = 1e9;
    int32_t splitFId = -1;
    int32_t splitCond = -1;

    const auto& linearL2Target = dynamic_cast<const LinearL2&>(target);

    MultiDimArray<2, float> splitScores({fCount_, totalCond_});

    // TODO can parallelize by totalBins
    parallelFor(0, fCount_, [&](int fId) {
        for (int cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
            for (auto &l : leaves_) {
                splitScores[fId][cond] += l->splitScore(linearL2Target, fId, cond);
            }
        }
    });

    for (int fId = 0; fId < fCount_; ++fId) {
        for (int cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
            float sScore = splitScores[fId][cond];
            if (sScore < bestScore) {
                bestScore = sScore;
                splitFId = fId;
                splitCond = cond;
            }
        }
    }

    if (splitFId < 0 || splitCond < 0) {
        throw std::runtime_error("Failed to find the best split");
    }

    std::cout << "best split: " << splitFId << " " << splitCond <<  std::endl;

    return std::make_pair(splitFId, splitCond);
}

void GreedyLinearObliviousTreeLearnerV2::initNewLeaves(GreedyLinearObliviousTreeLearnerV2::TSplit split) {
    newLeaves_.clear();

    for (auto& l : leaves_) {
        auto newLeavesPair = l->split(split.first, split.second);
        newLeaves_.emplace_back(newLeavesPair.first);
        newLeaves_.emplace_back(newLeavesPair.second);
    }

    int32_t splitFId = split.first;
    int32_t splitCond = split.second;

    float border = grid_->borders(splitFId).at(splitCond);
    auto fColumnRef = fColumnsRefs_[splitFId];

    for (int i = 0; i < (int)leaves_.size(); ++i) {
        samplesLeavesCnt_[2 * i] = 0;
        samplesLeavesCnt_[2 * i + 1] = 0;
    }

    parallelFor(0,nSamples_, [&](int i) {
        if (fColumnRef[i] <= border) {
            leafId_[i] = 2 * leafId_[i];
        } else {
            leafId_[i] = 2 * leafId_[i] + 1;
        }
        ++samplesLeavesCnt_[leafId_[i]];
    });

    for (int i = 0; i < leaves_.size(); ++i) {
        fullUpdate_[2 * i] = samplesLeavesCnt_[2 * i] <= samplesLeavesCnt_[2 * i + 1];
        fullUpdate_[2 * i + 1] = !fullUpdate_[2 * i];
    }
}

void GreedyLinearObliviousTreeLearnerV2::updateNewLeaves(
        const BinarizedDataSet& bds,
        const DataSet& ds,
        int oldNUsedFeatures,
        ConstVecRef<float> ys,
        ConstVecRef<float> ws) {
    int nUsedFeatures = usedFeaturesOrdered_.size();
    int nLeaves = newLeaves_.size();

    std::set<int> usedFeaturesSet(usedFeaturesOrdered_.begin(), usedFeaturesOrdered_.end());

    // for partial updates
    MultiDimArray<4, float> newCors({nThreads_, nLeaves, totalBins_, nUsedFeatures + 1});
    MultiDimArray<3, float> newXy({nThreads_, nLeaves, totalBins_});
    std::vector<std::vector<float>> curX(nThreads_, std::vector<float>(nUsedFeatures + 1, 0.));

    // for full update
    MultiDimArray<3, LinearL2Stat> stats({nThreads_, nLeaves, totalBins_}, nUsedFeatures + 1, nUsedFeatures);

    // 4) build full correlations only for left children, update new correlations for right ones

    TIME_BLOCK_START(ComputingFullCorrelationsAndNewCorrelations)

    parallelFor(0, nSamples_, [&](int blockId, int i) {
        auto& x = curX[blockId];
        ds.fillSample(i, usedFeaturesOrdered_, x);
        int lId = leafId_[i];
        auto bins = bds.sampleBins(i); // todo cache

        const float y = ys[i];
        const float w = ws[i];

        LinearL2StatOpParams params;

        if (fullUpdate_[lId]) {
            for (int fId = 0; fId < fCount_; ++fId) {
                int bin = (int)binOffsets_[fId] + bins[fId];
                params.vecAddMode = LinearL2StatOpParams::VecAddMode::FullCorrelation;
                stats[blockId][lId][bin].append(x.data(), y, w, params);
            }
        } else {
            if (nUsedFeatures > oldNUsedFeatures) {
                float fVal = x[oldNUsedFeatures];
                float wf = fVal * w;
                for (int fId = 0; fId < fCount_; ++fId) {
                    int bin = (int)binOffsets_[fId] + bins[fId];

                    for (int f = 0; f < oldNUsedFeatures; ++f) {
                        newCors[blockId][lId][bin][f] += x[f] * wf;
                    }
                    newCors[blockId][lId][bin][oldNUsedFeatures] += fVal * wf;
                    newXy[blockId][lId][bin] += y * wf;
                }
            }
        }
    });

    TIME_BLOCK_END(ComputingFullCorrelationsAndNewCorrelations)

    TIME_BLOCK_START(DoPartialUpdates)

    // for right leaves, prefix sum new correlations
    if (nUsedFeatures > oldNUsedFeatures) {
        // todo change order?
        parallelFor(0, fCount_, [&](int fId) {
            for (int localBinId = 0; localBinId <= (int)grid_->conditionsCount(fId); ++localBinId) {
                int bin = binOffsets_[fId] + localBinId;
                for (int lId = 0; lId < (int)newLeaves_.size(); ++lId) {
                    if (!fullUpdate_[lId]) {
                        for (int thId = 0; thId < nThreads_; ++thId) {
                            if (localBinId != 0) {
                                for (int i = 0; i <= oldNUsedFeatures; ++i) {
                                    newCors[thId][lId][bin][i] += newCors[thId][lId][bin - 1][i];
                                }
                                newXy[thId][lId][bin] += newXy[thId][lId][bin - 1];
                            }
                            LinearL2StatOpParams params;
                            params.shift = -1;
                            newLeaves_[lId]->stats_[bin].append(newCors[thId][lId][bin].data(),
                                                                newXy[thId][lId][bin], 1.0, params);
                        }
                    }
                }
            }
        });
    }

    TIME_BLOCK_END(DoPartialUpdates)

    TIME_BLOCK_START(FullUpdates)

    // For left leaves, sum up stats and then compute prefix sums
    parallelFor(0, totalBins_, [&](int bin) {
        for (int lId = 0; lId < (int)newLeaves_.size(); ++lId) {
            if (fullUpdate_[lId]) {
                for (int blockId = 0; blockId < nThreads_; ++blockId) {
                    newLeaves_[lId]->stats_[bin] += stats[blockId][lId][bin];
                }
            }
        }
    });

    parallelFor(0, newLeaves_.size(), [&](int lId) {
        if (fullUpdate_[lId]) {
            newLeaves_[lId]->prefixSumBins();
        }
    });

    TIME_BLOCK_END(FullUpdates)


    TIME_BLOCK_START(SubtractLeftsFromParents)
    // subtract lefts from parents to obtain inner parts of right children

    parallelFor(0, leaves_.size(), [&](int lId) {
        auto& parent = leaves_[lId];
        auto& left = newLeaves_[2 * lId];
        auto& right = newLeaves_[2 * lId + 1];

        // This - and += ops will only update inner correlations -- exactly what we need
        // new feature correlation will stay the same

        if (fullUpdate_[left->id_]) {
            for (int bin = 0; bin < totalBins_; ++bin) {
                right->stats_[bin] += parent->stats_[bin] - left->stats_[bin];
            }
        } else {
            for (int bin = 0; bin < totalBins_; ++bin) {
                left->stats_[bin] += parent->stats_[bin] - right->stats_[bin];
            }
        }
    });

    TIME_BLOCK_END(SubtractLeftsFromParents)
}
