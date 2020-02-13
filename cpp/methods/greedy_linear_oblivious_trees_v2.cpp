#include "greedy_linear_oblivious_trees_v2.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <chrono>

#include <core/vec_factory.h>
#include <core/matrix.h>

HistogramV2::HistogramV2(BinarizedDataSet& bds, GridPtr grid, unsigned int nUsedFeatures, int lastUsedFeatureId)
        : bds_(bds)
        , grid_(std::move(grid))
        , nUsedFeatures_(nUsedFeatures)
        , lastUsedFeatureId_(lastUsedFeatureId) {
    for (int i = 0; i < (int)grid_->totalBins(); ++i) {
        hist_.emplace_back(nUsedFeatures + 1, nUsedFeatures);
    }
}

void HistogramV2::addFullCorrelation(int bin, Vec x, double y) {
    hist_[bin].addFullCorrelation(x, y);
}

void HistogramV2::addNewCorrelation(int bin, const std::vector<double>& xtx, double xty) {
    hist_[bin].addNewCorrelation(xtx, xty);
}

void HistogramV2::addBinStat(int bin, const BinStat& stats) {
    hist_[bin] += stats;
}

void HistogramV2::prefixSumBins() {
    for (int fId = 0; fId < (int)grid_->nzFeaturesCount(); ++fId) {
        int offset = grid_->binOffsets()[fId];
        for (int localBinId = 1; localBinId <= grid_->conditionsCount(fId); ++localBinId) {
            int bin = offset + localBinId;
            hist_[bin] += hist_[bin - 1];
        }
    }
}

std::shared_ptr<Mx> HistogramV2::getW(double l2reg) {
    if (lastUsedFeatureId_ == -1) {
        throw std::runtime_error("No features are used");
    }

    uint32_t offset = grid_->binOffsets().at(lastUsedFeatureId_);
    uint32_t lastPos = offset + grid_->conditionsCount(lastUsedFeatureId_);

    auto XTX = hist_[lastPos].getXTX();
    auto XTy = hist_[lastPos].getXTy();

    Mx XTX_r = XTX + Diag(XTX.ydim(), l2reg);

    try {
        auto fullW = XTX_r.inverse() * XTy;
        auto wVec = fullW.slice(0, nUsedFeatures_);
        return std::make_shared<Mx>(wVec, wVec.size(), 1);
    } catch (...) {
        std::cout << "No inverse" << std::endl;
        return std::make_shared<Mx>((int64_t)XTX.ydim(), (int64_t)XTX.xdim());
    }
}

double HistogramV2::computeScore(Mx& XTX, Mx& XTy, double XTX_trace, uint32_t cnt, double l2reg,
                               double traceReg) {
    if (cnt == 0) {
        return 0;
    }

    try {
        Mx w = XTX.inverse() * XTy;
//    std::cout << "w is " << w << std::endl;

        Mx c1(XTy.T() * w);
        c1 *= -2;
        assert(c1.xdim() == 1 && c1.ydim() == 1);

        Mx c2(w.T() * XTX * w);
        assert(c2.xdim() == 1 && c2.ydim() == 1);

        Mx reg = w.T() * w * l2reg;
        assert(reg.xdim() == 1 && reg.ydim() == 1);

        Mx res = c1 + c2 + reg;

        return res.get(0, 0) + traceReg * XTX_trace / XTX.ydim();
    } catch (...) {
        std::cout << "No inverse" << std::endl;
        return 0;
    }
}

std::pair<double, double> HistogramV2::splitScore(int fId, int condId, double l2reg,
                                                double traceReg) {
    uint32_t offset = grid_->binOffsets()[fId];
    uint32_t binPos = offset + condId;
    uint32_t lastPos = offset + grid_->conditionsCount(fId);

    auto XTX_binPos = hist_[binPos].getXTX();
    auto XTX_lastPos = hist_[lastPos].getXTX();

    auto XTy_binPos = hist_[binPos].getXTy();
    auto XTy_lastPos = hist_[lastPos].getXTy();

    auto cnt_binPos = hist_[binPos].getCnt();
    auto cnt_lastPos = hist_[lastPos].getCnt();

    auto trace_binPos = hist_[binPos].getTrace();
    auto trace_lastPos = hist_[lastPos].getTrace();

    auto ydim = XTX_binPos.ydim();

    Mx left_XTX = XTX_binPos + Diag(ydim, l2reg);
    Mx right_XTX = XTX_lastPos - XTX_binPos + Diag(ydim, l2reg);

    Mx left_XTy(XTy_binPos);
    Mx right_XTy = XTy_lastPos - XTy_binPos;

    uint32_t left_cnt = cnt_binPos;
    uint32_t right_cnt = cnt_lastPos - cnt_binPos;

    double left_XTX_trace = trace_binPos;
    double right_XTX_trace = trace_lastPos - trace_binPos;

//    std::cout << "split fId: " << fId << ", cond: " << condId << " ";

    auto resLeft = computeScore(left_XTX, left_XTy, left_XTX_trace, left_cnt, l2reg, traceReg);
    auto resRight = computeScore(right_XTX, right_XTy, right_XTX_trace, right_cnt, l2reg, traceReg);

    return std::make_pair(resLeft, resRight);
}


void HistogramV2::printCnt() {
    if (lastUsedFeatureId_ == -1) {
        throw std::runtime_error("No features are used, can not print eig");
    }

    uint32_t offset = grid_->binOffsets().at(lastUsedFeatureId_);
    uint32_t lastPos = offset + grid_->conditionsCount(lastUsedFeatureId_);

    std::cout << "cnt: " << hist_[lastPos].getCnt() << std::endl;
}

void HistogramV2::printEig(Mx& M) {
    auto eigs = torch::eig(M.data().view({M.ydim(), M.xdim()}), false);
    auto vals = std::get<0>(eigs);
    for (int i = 0; i < (int)vals.size(0); ++i) {
        std::cout << vals[i].data()[0] << ", ";
    }
}

void HistogramV2::printEig(double l2reg) {
    if (lastUsedFeatureId_ == -1) {
        throw std::runtime_error("No features are used, can not print eig");
    }

    uint32_t offset = grid_->binOffsets().at(lastUsedFeatureId_);
    uint32_t lastPos = offset + grid_->conditionsCount(lastUsedFeatureId_);

    auto XTX = hist_[lastPos].getXTX();
    std::cout << "XTX: " << XTX << std::endl;

    auto XTy = hist_[lastPos].getXTy();
    std::cout << "XTy: " << XTy << std::endl;

    auto w = getW(l2reg);
    std::cout << "w: " << *w << std::endl;

    std::cout << "XTX eig: ";
    printEig(XTX);
    std::cout << std::endl;

    Mx XTX_Reg = XTX + Diag(XTX.xdim(), l2reg);

    std::cout << "(XTX + Diag(" << l2reg << ")) eig: ";
    printEig(XTX_Reg);
    std::cout << std::endl;

    std::cout << "XTX trace: " << hist_[lastPos].getTrace() << std::endl;
}

void HistogramV2::print() {
    std::cout << "Hist (nUsedFeatures=" << nUsedFeatures_ << ") {" << std::endl;
    for (int fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
        std::cout << "fId: " << fId << std::endl;
        for (int cond = 0; cond <= grid_->conditionsCount(fId); ++cond) {
            uint32_t offset = grid_->binOffsets().at(fId);
            uint32_t bin = offset + cond;
            std::cout << "fId: " << fId << ", cond: " << cond << ", XTX: " << hist_[bin].getXTX()
                    << ", XTy: " << hist_[bin].getXTy()
                    << ", cnt: " << hist_[bin].getCnt() << std::endl;
        }
    }
    std::cout << "}" << std::endl;
}

HistogramV2& HistogramV2::operator+=(const HistogramV2& h) {
    for (int bin = 0; bin < (int)grid_->totalBins(); ++bin) {
        hist_[bin] += h.hist_[bin];
    }

    return *this;
}

HistogramV2& HistogramV2::operator-=(const HistogramV2& h) {
    for (int bin = 0; bin < (int)grid_->totalBins(); ++bin) {
        hist_[bin] -= h.hist_[bin];
    }

    return *this;
}

HistogramV2 operator-(const HistogramV2& lhs, const HistogramV2& rhs) {
    HistogramV2 res(lhs);
    res -= rhs;

    return res;
}

HistogramV2 operator+(const HistogramV2& lhs, const HistogramV2& rhs) {
    HistogramV2 res(lhs);
    res += rhs;

    return res;
}


class LinearObliviousTreeLeafV2 : std::enable_shared_from_this<LinearObliviousTreeLeafV2> {
public:
    LinearObliviousTreeLeafV2(BinarizedDataSet& bds, GridPtr grid, double l2reg, double traceReg, unsigned int maxDepth,
            unsigned int nUsedFeatures, int lastUsedFeatureId)
            : bds_(bds)
            , grid_(std::move(grid))
            , l2reg_(l2reg)
            , traceReg_(traceReg)
            , maxDepth_(maxDepth)
            , nUsedFeatures_(nUsedFeatures)
            , lastUsedFeatureId_(lastUsedFeatureId) {
        hist_ = std::make_unique<HistogramV2>(bds_, grid_, nUsedFeatures, lastUsedFeatureId);
        id_ = 0;
    }

    double splitScore(int fId, int condId) {
        auto sScore = hist_->splitScore(fId, condId, l2reg_, traceReg_);
        return sScore.first + sScore.second;
    }

    void fit() {
        if (w_) {
            return;
        }
        w_ = hist_->getW(l2reg_);
    }

    double value(const Vec& x) {
        if (!w_) {
            throw std::runtime_error("Not fitted");
        }

        auto x_ref = x.arrayRef();

        std::vector<double> tmp;
        for (auto f : usedFeaturesInOrder_) {
            tmp.push_back(x_ref[f]);
        }
        auto xVec = VecFactory::fromVector(tmp);
        Mx X(xVec, xVec.size(), 1);

        Mx res = X.T() * (*w_);

        return res.get(0, 0);
    }

    bool isInRegion(const Vec& x) {
        auto xRef = x.arrayRef();
        for (auto& s : splits_) {
            int32_t fId = std::get<0>(s);
            int32_t condId = std::get<1>(s);
            bool isLeft = std::get<2>(s);

            int32_t origFId = grid_->nzFeatures().at(fId).origFeatureId_;
            float border = grid_->borders(fId).at(condId);

            if ((xRef[origFId] <= border) ^ isLeft) {
                return false;
            }
        }

        return true;
    }

    std::pair<std::shared_ptr<LinearObliviousTreeLeafV2>, std::shared_ptr<LinearObliviousTreeLeafV2>>
    split(int32_t fId, int32_t condId) {
        int origFId = grid_->origFeatureIndex(fId);
        unsigned int nUsedFeatures = nUsedFeatures_ + (1 - usedFeatures_.count(origFId));
//        std::cout << "new nUsedFeatures: " << nUsedFeatures << std::endl;

        auto left = std::make_shared<LinearObliviousTreeLeafV2>(bds_, grid_,
                l2reg_, traceReg_,
                maxDepth_, nUsedFeatures, fId);
        auto right = std::make_shared<LinearObliviousTreeLeafV2>(bds_, grid_,
                l2reg_, traceReg_,
                maxDepth_, nUsedFeatures, fId);

        initChildren(left, right, fId, condId);

        return std::make_pair(left, right);
    }

    void printHists() {
        hist_->print();
    }

    void printInfo() {
        hist_->printEig(l2reg_);
        hist_->printCnt();
        printSplits();
        std::cout << std::endl;
    }

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

    BinarizedDataSet& bds_;

    GridPtr grid_;
    std::set<int32_t> usedFeatures_;
    std::vector<int32_t> usedFeaturesInOrder_;
    std::shared_ptr<Mx> w_;
    std::vector<std::tuple<int32_t, int32_t, bool>> splits_;

    double l2reg_;
    double traceReg_;

    unsigned int maxDepth_;
    unsigned int nUsedFeatures_;
    int lastUsedFeatureId_;

    int32_t id_;

    std::unique_ptr<HistogramV2> hist_;
};



ModelPtr GreedyLinearObliviousTreeLearnerV2::fit(const DataSet& ds, const Target& target) {
    auto beginAll = std::chrono::steady_clock::now();

    auto tree = std::make_shared<LinearObliviousTreeV2>(grid_);

    // todo cache
    auto bds = cachedBinarize(ds, grid_);

    std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> leaves;

    std::vector<int32_t> leafId(ds.samplesCount(), 0);

    auto ys = target.targets().arrayRef();

    std::set<int64_t> usedFeatures;
    usedFeatures.insert(biasCol_);

    if (biasCol_ == -1) {
        // TODO
        throw std::runtime_error("provide bias col!");
    }

    std::cout << "start" << std::endl;

    auto beginC = std::chrono::steady_clock::now();
    cacheDs(ds);
    DataSet curDs(ds.sampleMx({biasCol_}), target.targets());
    auto endC = std::chrono::steady_clock::now();
    auto time_msC = std::chrono::duration_cast<std::chrono::milliseconds>(endC - beginC).count();
    std::cout << "caching time: " << time_msC << " [ms]" << std::endl;



    auto root = std::make_shared<LinearObliviousTreeLeafV2>(bds, this->grid_, l2reg_, traceReg_, maxDepth_ + 1, 1, -1);
    root->usedFeatures_.insert(biasCol_);
    root->usedFeaturesInOrder_.push_back(biasCol_);

    parallelFor(0, curDs.samplesCount(), [&](int blockId, int i) {
        Vec x = curDs.sample(i);
        auto bins = bds.sampleBins(i); // todo cache it somehow?
        double y = ys[i];

        for (int fId = 0; fId < fCount_; ++fId) {
            int offset = (int)binOffsets_[fId];
            int bin = offset + bins[fId];
            stats_[blockId][0][bin].addFullCorrelation(x, y);
        }
    });
//    }

    parallelFor(0, totalBins_, [&](int bin) {
        for (int blockId = 1; blockId < nThreads_; ++blockId) {
            stats_[0][0][bin] += stats_[blockId][0][bin];
        }
    });

    parallelFor(0, totalBins_, [&](int bin) {
        root->hist_->addBinStat(bin, stats_[0][0][bin]);
    });

    root->hist_->prefixSumBins();

    leaves.emplace_back(std::move(root));




    // Root is built





    for (unsigned int d = 0; d < maxDepth_; ++d) {
        // Update new correlations

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

//        std::cout << 1 << std::endl;

        auto nUsedFeatures = leaves[0]->nUsedFeatures_;

        parallelFor(0, nThreads_, [&](int i) {
            for (int j = 0; j < (int)leaves.size(); ++j) {
                for (int k = 0; k < totalBins_; ++k) {
                    for (int l = 0; l <= (int)nUsedFeatures; ++l) {
                        h_XTX_[i][j][k][l] = 0;
                    }
                    h_XTy_[i][j][k] = 0;
                }
            }
        });

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "1 in " << time_ms << " [ms]" << std::endl;

        begin = std::chrono::steady_clock::now();
//        std::cout << 2 << std::endl;

        parallelFor(0, bds.samplesCount(), [&](int blockId, int sampleId) {
            auto bins = bds.sampleBins(sampleId);
            Vec x = curDs.sample(sampleId);
            auto xRef = x.arrayRef();
            unsigned int lId = leafId[sampleId];

            for (int fId = 0; fId < fCount_; ++fId) {
                auto origFId = grid_->origFeatureIndex(fId);
                if (usedFeatures.count(origFId) != 0) continue;

                int bin = binOffsets_[fId] + bins[fId];

                double fVal = fColumnsRefs_[fId][sampleId];

                for (unsigned int i = 0; i < nUsedFeatures; ++i) {
                    h_XTX_[blockId][lId][bin][i] += xRef[i] * fVal;
                }
                h_XTX_[blockId][lId][bin][nUsedFeatures] += fVal * fVal;
                h_XTy_[blockId][lId][bin] += fVal * ys[sampleId];
            }
//        }
        });

        std::cout << 2.5 << std::endl;

        // todo change order?
        parallelFor(0, fCount_, [&](int fId) {
            auto origFId = grid_->origFeatureIndex(fId);
            if (usedFeatures.count(origFId) != 0) return;

            for (int localBinId = 0; localBinId <= (int)grid_->conditionsCount(fId); ++localBinId) {
                int bin = binOffsets_[fId] + localBinId;
                for (int lId = 0; lId < (int)leaves.size(); ++lId) {
                    for (int thId = 0; thId < nThreads_; ++thId) {
                        if (localBinId != 0) {
                            for (unsigned int i = 0; i <= nUsedFeatures; ++i) {
                                h_XTX_[thId][lId][bin][i] += h_XTX_[thId][lId][bin - 1][i];
                            }
                            h_XTy_[thId][lId][bin] += h_XTy_[thId][lId][bin - 1];
                        }
                        leaves[lId]->hist_->addNewCorrelation(bin, h_XTX_[thId][lId][bin], h_XTy_[thId][lId][bin]);
                    }
                }
            }
        });

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "2 in " << time_ms << " [ms]" << std::endl;



        // Find best split





        begin = std::chrono::steady_clock::now();

        double bestSplitScore = 1e9;
        int32_t splitFId = -1;
        int32_t splitCond = -1;

        std::vector<std::vector<double>> splitScores;
        for (int fId = 0; fId < fCount_; ++fId) {
            std::vector<double> fSplitScores;
            fSplitScores.resize(grid_->conditionsCount(fId), 0.0);
            splitScores.emplace_back(std::move(fSplitScores));
        }

        parallelFor(0, fCount_, [&](int fId) {
//        for (int fId = 0; fId < fCount_; ++fId) {
//            for (int cond = 0; cond < (int) grid_->conditionsCount(fId); ++cond) {
            parallelFor<1>(0, grid_->conditionsCount(fId), [&](int cond) {
                for (auto &l : leaves) {
                    splitScores[fId][cond] += l->splitScore(fId, cond);
                }
            });
//            }
//        }
        });

        for (int fId = 0; fId < fCount_; ++fId) {
            for (int64_t cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
                double sScore = splitScores[fId][cond];
//                std::cout << "fId: " << fId << ", cond: " << cond << std::setprecision(6) << ", split score: " << sScore << std::endl;
                if (sScore < bestSplitScore) {
                    bestSplitScore = sScore;
                    splitFId = fId;
                    splitCond = cond;
                }
            }
        }

        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Best split found in " << time_ms << "[ms], splitting" << std::endl;

//        std::cout << "Best split fId = " << splitFId << ", split cond = " << splitCond << std::endl;




        // Split





        begin = std::chrono::steady_clock::now();

        // 1) find new leaf ids

//        std::cout << "1)" << std::endl;

        double border = grid_->borders(splitFId).at(splitCond);
        auto fColumnRef = fColumnsRefs_[splitFId];

        parallelFor(0, ds.samplesCount(), [&](int i) {
            if (fColumnRef[i] <= border) {
                leafId[i] = 2 * leafId[i];
            } else {
                leafId[i] = 2 * leafId[i] + 1;
            }
        });

        // 2) init new leaves

//        std::cout << "2)" << std::endl;

        std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> newLeaves;
        for (int i = 0; i < (int)leaves.size() * 2; ++i) {
            newLeaves.emplace_back(std::shared_ptr<LinearObliviousTreeLeafV2>(nullptr));
        }

        parallelFor(0, leaves.size(), [&](int lId) {
            auto& l = leaves[lId];
            auto splits = l->split(splitFId, splitCond);
            newLeaves[splits.first->id_] = std::move(splits.first);
            newLeaves[splits.second->id_] = std::move(splits.second);
        });

        // 3) update current ds, reset stats

//        std::cout << "3)" << std::endl;

        int32_t splitOrigFId = grid_->origFeatureIndex(splitFId);
        if (usedFeatures.count(splitOrigFId) == 0) {
            curDs.addColumn(fColumns_[splitFId]);
            usedFeatures.insert(splitOrigFId);
        }
        nUsedFeatures = usedFeatures.size();

        parallelFor(0, nThreads_, [&](int i) {
            for (int j = 0; j < (int)newLeaves.size(); ++j) {
                for (int k = 0; k < totalBins_; ++k) {
                    for (int l = 0; l <= (int)nUsedFeatures; ++l) {
                        h_XTX_[i][j][k][l] = 0;
                    }
                    h_XTy_[i][j][k] = 0;
                    stats_[i][j][k].reset();
                    stats_[i][j][k].setFilledSize(nUsedFeatures);
                }
            }
        });

        nUsedFeatures = usedFeatures.size();

        // 4) build full correlations for left AND right children
        // TODO build only for left

        parallelFor(0, curDs.samplesCount(), [&](int blockId, int i) {
            Vec x = curDs.sample(i);
            int lId = leafId[i];
            auto bins = bds.sampleBins(i); // todo cache
            double y = ys[i];

            for (int fId = 0; fId < fCount_; ++fId) {
                int bin = (int)binOffsets_[fId] + bins[fId];
                stats_[blockId][lId][bin].addFullCorrelation(x, y);
            }
        });

//        std::cout << "4.1)" << std::endl;

        parallelFor(0, totalBins_, [&](int bin) {
            for (int lId = 0; lId < (int)newLeaves.size(); ++lId) {
                for (int blockId = 1; blockId < nThreads_; ++blockId) {
                    stats_[0][lId][bin] += stats_[blockId][lId][bin];
                }
            }
        });

//        std::cout << "4.2)" << std::endl;

        parallelFor(0, totalBins_, [&](int bin) {
            for (int lId = 0; lId < (int)newLeaves.size(); ++lId) {
                newLeaves[lId]->hist_->addBinStat(bin, stats_[0][lId][bin]);
            }
        });

        parallelFor(0, newLeaves.size(), [&](int lId) {
            newLeaves[lId]->hist_->prefixSumBins();
        });

//     TODO 5) subtract lefts from parents to obtain rights


        end = std::chrono::steady_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Split done in" << time_ms << "[ms]" << std::endl;

        leaves = std::move(newLeaves);
    }

    parallelFor(0, leaves.size(), [&](int lId) {
        auto& l = leaves[lId];
        l->fit();
    });

    tree->leaves_ = std::move(leaves);

    auto endAll = std::chrono::steady_clock::now();
    auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endAll - beginAll).count();
    std::cout << "ALl fit done in " << time_ms << "[ms]" << std::endl;

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
    binOffsets_ = grid_->binOffsets();
    fCount_ = grid_->nzFeaturesCount();

    nThreads_ = (int)GlobalThreadPool<0>().numThreads();
    for (int i = 0; i < nThreads_; ++i) { // threads
        h_XTX_.emplace_back();
        h_XTy_.emplace_back();
        stats_.emplace_back();
        for (int j = 0; j < (1 << maxDepth_); ++j) { // leaves
            h_XTy_[i].emplace_back(totalBins_, 0.0);
            stats_[i].emplace_back();
            h_XTX_[i].emplace_back();
            for (int k = 0; k < totalBins_; ++k) { // bins
                h_XTX_[i][j].emplace_back(grid_->nzFeaturesCount() + 2, 0.0);
                stats_[i][j].emplace_back(BinStat((int)grid_->nzFeaturesCount() + 2, 1));
            }
        }
    }

    isDsCached_ = true;
}


double LinearObliviousTreeV2::value(const Vec& x) const {
    for (auto& l : leaves_) {
        if (l->isInRegion(x)) {
            return scale_ * l->value(x);
        }
    }

    throw std::runtime_error("given x does not belong to any region O_o");
}

void LinearObliviousTreeV2::appendTo(const Vec &x, Vec to) const {
    to += static_cast<const LinearObliviousTreeV2*>(this)->value(x);
}

double LinearObliviousTreeV2::value(const Vec &x) {
    return static_cast<const LinearObliviousTreeV2*>(this)->value(x);
}

void LinearObliviousTreeV2::grad(const Vec &x, Vec to) {
    throw std::runtime_error("Unimplemented");
}
