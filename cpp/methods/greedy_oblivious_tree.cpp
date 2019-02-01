#include "greedy_oblivious_tree.h"
#include <core/vec.h>
#include <core/buffer.h>
#include <data/histogram.h>
#include <data/grid.h>
#include <data/binarized_dataset.h>
#include <targets/l2.h>
#include <models/oblivious_tree.h>
#include <util/parallel_executor.h>
namespace {

    struct DataPartition {
        int64_t Offset = 0;
        int64_t Size = 0;
    };


    template <class StatBasedTarget>
    class Subsets {
    public:
        using Stat = typename StatBasedTarget::AdditiveStat;

        Subsets(const StatBasedTarget& target,
                const BinarizedDataSet& ds)
            : ds_(ds) {

            target.makeStats(&stat_, &indices_);
            bins_ = Buffer<int32_t>(stat_.size());
            leaves_.push_back({0, indices_.size()});
            updateLeavesStats();
        }

        void split(const BinaryFeature& feature) {
            ArrayRef<int32_t> binsRef = bins_.arrayRef();
            auto indicesRef = indices_.arrayRef();
            auto statRef = stat_.arrayRef();

            ds_.visitFeature(feature.featureId_,
                             indicesRef,
                             [&](int32_t i, uint8_t bin) {
                binsRef[i] |= (bin  > feature.conditionId_) << level_;
            });
            ++level_;

            std::vector<DataPartition> newLeaves;
            newLeaves.resize(1 << level_);

            for (int32_t bin : binsRef) {
                newLeaves[bin].Size++;
            }

            std::vector<int64_t> writeOffsets(newLeaves.size());
            for (int32_t i =  1; i < newLeaves.size(); ++i) {
                newLeaves[i].Offset = newLeaves[i - 1].Offset + newLeaves[i - 1].Size;
                writeOffsets[i] = newLeaves[i].Offset;
            }

            //reorder data now

            Buffer<Stat> nextStat(stat_.size());
            Buffer<int32_t> nextIndices(indices_.size());
            Buffer<int32_t> nextBins(bins_.size());

            auto nextStatRef = nextStat.arrayRef();
            auto nextIndicesRef = nextIndices.arrayRef();
            auto nextBinsRef = nextBins.arrayRef();


            for (uint32_t i = 0; i < binsRef.size(); ++i) {
                int64_t writeOffset = writeOffsets[binsRef[i]]++;

                nextBinsRef[writeOffset] = binsRef[i];
                nextIndicesRef[writeOffset] = indicesRef[i];
                nextStatRef[writeOffset] = statRef[i];
            }

            stat_.Swap(nextStat);
            indices_.Swap(nextIndices);
            bins_.Swap(nextBins);
            leaves_.swap(newLeaves);

            updateLeavesStats();

            prevHistograms_ = std::move(histograms_);
            histograms_.reset(nullptr);
        }

        template <class Visitor>
        void visitSplits(Visitor&& visitor) {
            buildHists();
            auto binFeatureOffsets = ds_.grid().binFeatureOffsets();
            auto binOffsets = ds_.binOffsets();
            auto leaves_stats_ref = leaves_stats_.arrayRef();

            for (int64_t leaf = 0; leaf < (1 << level_); ++leaf) {
                auto leafHistogram = histograms_->arrayRef().slice(leaf * ds_.totalBins(), ds_.totalBins());

                const auto nzFeaturesCount = ds_.grid().nzFeaturesCount();
                for (int64_t f = 0; f < nzFeaturesCount; ++f) {
                    const int32_t conditions =  ds_.grid().conditionsCount(f);
                    int32_t bins = conditions + 1;

                    const auto firstBin = binOffsets[f];
                    const auto seqCondition = binFeatureOffsets[f];
                    auto featureHistogram = leafHistogram.slice(firstBin, bins);
                    auto right = leaves_stats_ref[leaf];
                    Stat left;
                    for (int32_t bin = 0; bin < conditions; ++bin) {
                        left += featureHistogram[bin];
                        right -= featureHistogram[bin];
                        visitor(seqCondition + bin, left, right);
                    }
                }
            }
        }


        template <class IncrementCalcer>
        Vec bestIncrements(IncrementCalcer&& calcer) const {
            Vec leaves(leaves_.size());

            auto vals = leaves.arrayRef();
            auto sourceStat = leaves_stats_.arrayRef();
            for  (int32_t i = 0; i < vals.size(); ++i) {
                vals[i] = calcer(sourceStat[i]);
            }
            return leaves;
        }


    private:

        void buildHists() {
            if (histograms_ != nullptr) {
                return;
            }

            histograms_.reset(new Buffer<Stat>((1 << level_) * ds_.totalBins()));

            std::vector<int32_t> partsToBuild;
//
            if (prevHistograms_ != nullptr) {
                for (int32_t i = 0; i < 1 << (level_ - 1); ++i) {
                    int32_t leftPart = i;
                    int32_t rightPart = i | (1 <<  (level_ - 1));
                    if (leaves_[leftPart].Size < leaves_[rightPart].Size) {
                        partsToBuild.push_back(leftPart);
                    } else {
                        partsToBuild.push_back(rightPart);
                    }
                }
            } else {
                partsToBuild.resize(1 << level_);
                std::iota(partsToBuild.begin(), partsToBuild.end(), 0);
            }

            buildHistogramsForParts(partsToBuild, histograms_->arrayRef());

            if (prevHistograms_ != nullptr) {


                auto histograms = histograms_->arrayRef();
                auto prevHistograms = prevHistograms_->arrayRef();

                for (int32_t i = 0; i < 1 << (level_ - 1); ++i) {
                    int32_t leftPart = i;
                    int32_t rightPart = i | (1 <<  (level_ - 1));

                    for (int64_t bin = 0; bin < ds_.totalBins(); ++bin) {
                        auto minPartStat = histograms[ds_.totalBins() * i + bin];
                        auto prevStat = prevHistograms[ds_.totalBins() * i + bin];
                        auto maxPartStat = prevStat - minPartStat;

                        if (leaves_[leftPart].Size < leaves_[rightPart].Size) {
                            histograms[ds_.totalBins() * leftPart + bin] = minPartStat;
                            histograms[ds_.totalBins() * rightPart + bin] = maxPartStat;
                        } else {
                            histograms[ds_.totalBins() * rightPart + bin] = minPartStat;
                            histograms[ds_.totalBins() * leftPart + bin] = maxPartStat;
                        }
                    }
                }

                prevHistograms_.reset();
            }

        }

        void buildHistogramsForParts(ConstArrayRef<int32_t> partIds, ArrayRef<Stat> dst) const {
            auto& threadPool = GlobalThreadPool();
            for (int32_t i = 0; i < partIds.size(); ++i) {
                const int32_t partId = partIds[i];
                const auto& part = leaves_[partId];
                ConstArrayRef<int32_t> indices = indices_.arrayRef().slice(part.Offset, part.Size);
                ConstArrayRef<Stat> stat = stat_.arrayRef().slice(part.Offset, part.Size);

                ds_.visitGroups([dst, this, indices, stat, i, &threadPool](
                    FeaturesBundle bundle,
                    ConstArrayRef<uint8_t> data) {

//                    const int64_t minIndicesToBuildParallel = 4096;
                    auto binOffsets = ds_.binOffsets().slice(bundle.firstFeature_, bundle.groupSize());
                    ArrayRef<Stat> dstForBundle = dst.slice(ds_.totalBins() * i, ds_.totalBins());

                    threadPool.enqueue([=]() {
                        buildHistograms(bundle.groupSize(),
                                        stat,
                                        indices,
                                        binOffsets,
                                        data,
                                        dstForBundle
                        );
                    });
                });
            };
            threadPool.waitComplete();
        }

        void updateLeavesStats() {
            leaves_stats_ = Buffer<Stat>(leaves_.size());
            auto binsRef = bins_.arrayRef();
            auto statRef = stat_.arrayRef();
            auto leaves_stats_ref = leaves_stats_.arrayRef();
            for (uint32_t i = 0; i < binsRef.size(); ++i) {
                leaves_stats_ref[binsRef[i]] += statRef[i];
            }
        }
    private:

        const BinarizedDataSet& ds_;

        Buffer<Stat> stat_;
        Buffer<int32_t> indices_;
        Buffer<int32_t> bins_;

        Buffer<Stat> leaves_stats_;
        std::vector<DataPartition> leaves_;
        int32_t level_ = 0;

        UniquePtr<Buffer<Stat>> prevHistograms_;
        UniquePtr<Buffer<Stat>> histograms_;
    };
}



ModelPtr GreedyObliviousTree::fit(const DataSet& dataSet,
                                  const Target& target) {
    const auto& binarized = cachedBinarize(dataSet, grid_);


    using StatBasedTarget = StatBasedLoss<L2Stat>;
    const auto& l2Target = dynamic_cast<const StatBasedTarget&>(target);
    Subsets<StatBasedTarget> subsets(l2Target,
                                     binarized);

    double currentScore = 0;

    std::vector<BinaryFeature> splits;

    std::vector<double> scores(grid_->binFeaturesCount());

    for (int32_t depth = 0; depth < maxDepth_; ++depth) {
        std::fill(scores.begin(), scores.end(), 0);
        subsets.visitSplits([&](int32_t conditionIdx, const L2Stat& left, const L2Stat& right) {
            scores[conditionIdx] += l2Target.score(left) + l2Target.score(right);
        });

        double bestScore = 0;
        int32_t bestIdx = -1;

        for (int32_t i = 0; i < scores.size(); ++i) {
            if (scores[i] < bestScore) {
                bestScore = scores[i];
                bestIdx = i;
            }
        };

        if (bestScore < currentScore) {
            currentScore = bestScore;
        } else {
            break;
        }

        auto bestSplit = grid_->binFeature(bestIdx);
        splits.push_back(bestSplit);
        subsets.split(bestSplit);
    }

    auto leaves = subsets.bestIncrements([&](const L2Stat& stat) -> float {
        return static_cast<float>(l2Target.bestIncrement(stat));
    });

    return std::make_shared<ObliviousTree>(grid_, splits, leaves);
}
