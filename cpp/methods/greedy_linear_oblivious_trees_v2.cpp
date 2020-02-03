#include "greedy_linear_oblivious_trees_v2.h"

#include <memory>
#include <set>
#include <stdexcept>
#include <chrono>

#include <core/vec_factory.h>
#include <core/matrix.h>

HistogramV2::HistogramV2(BinarizedDataSet& bds, GridPtr grid, unsigned int histSize, unsigned int nUsedFeatures,
        int lastUsedFeatureId)
        : bds_(bds)
        , grid_(std::move(grid))
        , histSize_(histSize)
        , nUsedFeatures_(nUsedFeatures)
        , lastUsedFeatureId_(lastUsedFeatureId) {
    for (int i = 0; i < (int)grid_->totalBins(); ++i) {
        hist_XTX_.emplace_back(histSize, histSize);
        hist_XTy_.emplace_back(histSize, 1);
        hist_XTX_trace_.emplace_back(0);
        hist_cnt_.emplace_back(0);
    }
}

void HistogramV2::build(const DataSet& ds, const std::vector<int32_t>& indices) {
    auto ys = ds.target();
    auto ys_ref = ys.arrayRef();

    parallelFor<123321>(0, grid_->nzFeaturesCount(), [&](int fId) {
        int offset = bds_.binOffsets()[fId];
        bds_.visitFeature(fId, indices, [&](int i, int8_t localBinId) {
            int xId = indices[i];

//            std::cout << "fId = " << fId << ", localBinId = " << (int)localBinId << ", xId = " << xId << std::endl;
            int bin = offset + localBinId;

            auto XTX_ref = hist_XTX_[bin].arrayRef();

            auto x = ds.sample(xId);
            auto x_ref = x.arrayRef();

            // -- XTX
            for (unsigned int r = 0; r < x.size(); ++r) {
                for (unsigned int c = r; c < x.size(); ++c) {
                    unsigned int mx_idx = r * histSize_ + c;
                    float val = x_ref[r] * x_ref[c];
                    XTX_ref[mx_idx] += val;
                    if (c != r) {
                        mx_idx = c * histSize_ + r;
                        XTX_ref[mx_idx] += val;
                    }
                }
            }

            // --- XTy
            auto XTy_ref = hist_XTy_[bin].arrayRef();
            for (unsigned int r = 0; r < x.size(); ++r) {
                XTy_ref[r] += x_ref[r] * ys_ref[xId];
            }

            // --- cnt
            hist_cnt_[bin] += 1;
        });
    });
}

void HistogramV2::updateBin(int64_t fId, int64_t binId, const Vec& x, double y, double fVal, int corOffset) {
    int offset = bds_.binOffsets()[fId];
    int binPos = offset + (int)binId;

    auto XTX_ref = hist_XTX_[binPos].arrayRef();
    auto x_ref = x.arrayRef();

    unsigned int updatePos = nUsedFeatures_ + corOffset;

    unsigned int r = 0;
    unsigned int c = updatePos;

    for (unsigned int i = 0; i < 2 * updatePos + 1; ++i) {
        float updateVal;
        unsigned int x_coord = std::min(r, c);
        if (r != c) {
            updateVal = x_ref[x_coord] * (float)fVal;
        } else {
            updateVal = (float)(fVal * fVal);
        }
        unsigned int xtxPos = r * histSize_ + c;
        XTX_ref[xtxPos] += updateVal;

        if (r < c) {
            r++;
            if (r == c) {
                r = updatePos;
                c = 0;
            }
        } else if (c < r) {
            c++;
        }
    }

    auto XTy_ref = hist_XTy_[binPos].arrayRef();
    XTy_ref[updatePos] += (float)(fVal * y);
}

void HistogramV2::prefixSumBins() {
    parallelFor<123321>(0, grid_->nzFeaturesCount(), [&](int fId) {
        for (int localBinId = 1; localBinId <= (int)grid_->conditionsCount(fId); ++localBinId) {
            int offset = bds_.binOffsets()[fId];
            int binPos = offset + localBinId;

            auto hist_XTX_cur_ref = hist_XTX_[binPos].arrayRef();
            auto hist_XTy_cur_ref = hist_XTy_[binPos].arrayRef();

            auto hist_XTX_prev_ref = hist_XTX_[binPos - 1].arrayRef();
            auto hist_XTy_prev_ref = hist_XTy_[binPos - 1].arrayRef();

            for (unsigned int r = 0; r < nUsedFeatures_; ++r) {
                for (unsigned int c = 0; c < nUsedFeatures_; ++c) {
                    unsigned int mx_pos = r * histSize_ + c;
                    hist_XTX_cur_ref[mx_pos] += hist_XTX_prev_ref[mx_pos];
                }
                hist_XTy_cur_ref[r] += hist_XTy_prev_ref[r];
            }

            hist_cnt_[binPos] += hist_cnt_[binPos - 1];
//            hist_XTX_trace_[binPos] += hist_XTX_trace_[binPos - 1];
        }
    });
}

void HistogramV2::prefixSumBinsLastFeature(int corOffset) {
    parallelFor<123321>(0, grid_->nzFeaturesCount(), [&](int fId) {
        for (int localBinId = 1; localBinId <= (int)grid_->conditionsCount(fId); ++localBinId) {
            int offset = bds_.binOffsets()[fId];
            int binPos = offset + localBinId;

            auto hist_XTX_cur_ref = hist_XTX_[binPos].arrayRef();
            auto hist_XTy_cur_ref = hist_XTy_[binPos].arrayRef();

            auto hist_XTX_prev_ref = hist_XTX_[binPos - 1].arrayRef();
            auto hist_XTy_prev_ref = hist_XTy_[binPos - 1].arrayRef();

            unsigned int updatePos = nUsedFeatures_ + corOffset;

            unsigned int r = 0;
            unsigned int c = updatePos;

            for (unsigned int i = 0; i < 2 * updatePos + 1; ++i) {
                unsigned int mx_pos = r * histSize_ + c;
                hist_XTX_cur_ref[mx_pos] += hist_XTX_prev_ref[mx_pos];

                if (r < c) {
                    r++;
                    if (r == c) {
                        r = updatePos;
                        c = 0;
                    }
                } else if (c < r) {
                    c++;
                }
            }

            hist_XTy_cur_ref[updatePos] += hist_XTy_prev_ref[updatePos];

            // Don't update cnt here.

            // TODO trace
//            hist_XTX_trace_[binPos] += hist_XTX_trace_[binPos - 1];
        }
    });
}

std::shared_ptr<Mx> HistogramV2::getW(double l2reg) {
    if (lastUsedFeatureId_ == -1) {
        throw std::runtime_error("No features are used");
    }

    uint32_t offset = grid_->binOffsets().at(lastUsedFeatureId_);
    uint32_t lastPos = offset + grid_->conditionsCount(lastUsedFeatureId_);

    int xtx_dim = hist_XTX_[lastPos].ydim();

    Mx XTX = hist_XTX_[lastPos] + Diag(xtx_dim, l2reg);
    Mx& XTy = hist_XTy_[lastPos];

    try {
        auto fullW = XTX.inverse() * XTy;
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

    // Dealing with matrix singularity
    // Not sure what we should do in this case...
    // For now just return 0.
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

    int xtx_dim = hist_XTX_[binPos].ydim();

    Mx left_XTX = hist_XTX_[binPos] + Diag(xtx_dim, l2reg);
    Mx right_XTX = hist_XTX_[lastPos] - hist_XTX_[binPos] + Diag(xtx_dim, l2reg);

    Mx left_XTy(hist_XTy_[binPos]);
    Mx right_XTy = hist_XTy_[lastPos] - hist_XTy_[binPos];

    uint32_t left_cnt = hist_cnt_[binPos];
    uint32_t right_cnt = hist_cnt_[lastPos] - hist_cnt_[binPos];

    double left_XTX_trace = hist_XTX_trace_[binPos];
    double right_XTX_trace = hist_XTX_trace_[lastPos] - hist_XTX_trace_[binPos];

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

    std::cout << "cnt: " << hist_cnt_[lastPos] << std::endl;
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

    Mx& XTX = hist_XTX_[lastPos];
    std::cout << "XTX: " << XTX << std::endl;

    Mx& XTy = hist_XTy_[lastPos];
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

    std::cout << "XTX trace: " << hist_XTX_trace_[lastPos] << std::endl;
}

void HistogramV2::print() {
    std::cout << "Hist (nUsedFeatures=" << nUsedFeatures_ << ") {" << std::endl;
    for (int fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
        std::cout << "fId: " << fId << std::endl;
        for (int cond = 0; cond <= grid_->conditionsCount(fId); ++cond) {
            uint32_t offset = grid_->binOffsets().at(fId);
            uint32_t bin = offset + cond;
            std::cout << "fId: " << fId << ", cond: " << cond << ", XTX: " << hist_XTX_[bin] << ", XTy: " << hist_XTy_[bin] << ", cnt: " << hist_cnt_[bin] << std::endl;
        }
    }
    std::cout << "}" << std::endl;
}

HistogramV2 operator-(const HistogramV2& lhs, const HistogramV2& rhs) {
    GridPtr grid = lhs.grid_;

    HistogramV2 res(rhs.bds_, grid, lhs.histSize_, rhs.nUsedFeatures_, rhs.lastUsedFeatureId_);

    unsigned int nFeaturesToUse = lhs.nUsedFeatures_;

    for (uint32_t bin = 0; bin < grid->totalBins(); ++bin) {
        auto lhs_XTX_ref = lhs.hist_XTX_[bin].arrayRef();
        auto rhs_XTX_ref = rhs.hist_XTX_[bin].arrayRef();
        auto res_XTX_ref = res.hist_XTX_[bin].arrayRef();

        // --- XTX

        for (unsigned int i = 0; i < nFeaturesToUse; ++i) {
            for (unsigned int j = 0; j < nFeaturesToUse; ++j) {
                unsigned int pos = i * lhs.histSize_ + j;
                res_XTX_ref[pos] = lhs_XTX_ref[pos] - rhs_XTX_ref[pos];
            }
        }

        // --- XTy

        auto lhs_XTy_ref = lhs.hist_XTy_[bin].arrayRef();
        auto rhs_XTy_ref = rhs.hist_XTy_[bin].arrayRef();
        auto res_XTy_ref = res.hist_XTy_[bin].arrayRef();

        for (unsigned int i = 0; i < nFeaturesToUse; ++i) {
            res_XTy_ref[i] = lhs_XTy_ref[i] - rhs_XTy_ref[i];
        }

        // --

        res.hist_cnt_[bin] = lhs.hist_cnt_[bin] - rhs.hist_cnt_[bin];

        // --

        // TODO trace
    }

    res.lastUsedFeatureId_ = rhs.lastUsedFeatureId_;

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
        hist_ = std::make_unique<HistogramV2>(bds_, grid_, maxDepth, nUsedFeatures, lastUsedFeatureId);
        id_ = 0;
    }

    void updateBin(int64_t fId, int64_t binId, const Vec& x, double y, double f, int corOffset) {
        hist_->updateBin(fId, binId, x, y, f, corOffset);
    }

    double splitScore(int fId, int condId) {
        auto sScore = hist_->splitScore(fId, condId, l2reg_, traceReg_);
        return sScore.first + sScore.second;
    }

    void fit(const DataSet& ds) {
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
        for (auto& s : splits_) {
            int32_t fId = std::get<0>(s);
            int32_t condId = std::get<1>(s);
            bool isLeft = std::get<2>(s);

            int32_t origFId = grid_->nzFeatures().at(fId).origFeatureId_;
            float border = grid_->borders(fId).at(condId);

            if ((x(origFId) <= border) ^ isLeft) {
                return false;
            }
        }

        return true;
    }

    std::pair<std::shared_ptr<LinearObliviousTreeLeafV2>, std::shared_ptr<LinearObliviousTreeLeafV2>>
    split(const DataSet& curDs, Vec& newCol, std::vector<int32_t>& leafId, int32_t fId, int32_t condId) {
        int origFId = grid_->origFeatureIndex(fId);
        unsigned int nUsedFeatures = nUsedFeatures_ + (1 - usedFeatures_.count(origFId));
//        std::cout << "new nUsedFeatures: " << nUsedFeatures << std::endl;

        auto left = std::make_shared<LinearObliviousTreeLeafV2>(bds_, grid_,
                l2reg_, traceReg_,
                maxDepth_, nUsedFeatures, fId);
        auto right = std::make_shared<LinearObliviousTreeLeafV2>(bds_, grid_,
                l2reg_, traceReg_,
                maxDepth_, nUsedFeatures, fId);

        initChildren(curDs, newCol, leafId, left, right, fId, condId);

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
    void initChildren(const DataSet& curDs,
                      Vec& newCol, std::vector<int32_t>& leafId,
                      std::shared_ptr<LinearObliviousTreeLeafV2>& left,
                      std::shared_ptr<LinearObliviousTreeLeafV2>& right,
                      int32_t splitFId, int32_t condId) {
        left->id_ = 2 * id_;
        right->id_ = 2 * id_ + 1;

//        std::cout << "Splitting on feature " << origFeatureId << std::endl;

        std::vector<int32_t> leftXIds;
        std::vector<int32_t> rightXIds;

        auto newCol_ref = newCol.arrayRef();

        double border = grid_->borders(splitFId).at(condId);
        for (int i = 0; i < (int)newCol.size(); ++i) {
            if (leafId[i] == id_) {
                auto x = newCol_ref[i];
                if (x <= border) {
                    leafId[i] = left->id_;
                    leftXIds.push_back(i);
                } else {
                    leafId[i] = right->id_;
                    rightXIds.push_back(i);
                }
            }
        }

        // First, fully build left child, this takes three stages

        auto ys = curDs.target().arrayRef();

        int32_t origFeatureId = grid_->origFeatureIndex(splitFId);

        // 1) fill top corners with previous feature correlations

//        std::cout << "build left" << std::endl;
        left->hist_->build(curDs, leftXIds);

        // 2) update histograms with the new feature, but only if this feature is new

        if (usedFeatures_.count(origFeatureId) == 0) {
            parallelFor<345>(0, grid_->nzFeaturesCount(), [&](int fId) {
                bds_.visitFeature(fId, leftXIds, [&](int i, int8_t localBinId) {
                    int xId = leftXIds[i];
                    Vec x = curDs.sample(xId);
                    double fVal = newCol_ref[xId];
                    left->hist_->updateBin(fId, localBinId, x, ys[xId], fVal, -1);
                });
            });
        }

        // 3) prefix sum bins

        left->hist_->prefixSumBins();

//        // 1) fill top corners with previous feature correlations
//
////        std::cout << "build right" << std::endl;
//        right->hist_->build(curDs, rightXIds);
//
//        // 2) update histograms with the new feature, but only if this feature is new
//
//        if (usedFeatures_.count(origFeatureId) == 0) {
//            for (int32_t fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
//                bds_.visitFeature(fId, rightXIds, [&](int i, int8_t localBinId) {
//                    int xId = rightXIds[i];
//                    Vec x = curDs.sample(xId);
//                    double fVal = newCol_ref[xId];
//                    right->hist_->updateBin(fId, localBinId, x, ys[xId], fVal, -1);
//                });
//            }
//        }
//
//        // 3) prefix sum bins
//
//        right->hist_->prefixSumBins();

        // Now, build right child, this process also takes three steps...

        // 1) init first nUsedFeatures, using operator-

        right->hist_ = std::make_unique<HistogramV2>(*hist_ - *left->hist_);

        // 2) update with the last added feature, but only if this feature is new
        // TODO copy-paste -- this one is the same as for the left child

        if (usedFeatures_.count(origFeatureId) == 0) {
            parallelFor<345>(0, grid_->nzFeaturesCount(), [&](int fId) {
                bds_.visitFeature(fId, rightXIds, [&](int i, int8_t localBinId) {
                    int xId = rightXIds[i];
                    auto x = curDs.sample(xId);
                    auto fVal = newCol_ref[xId];
                    right->hist_->updateBin(fId, localBinId, x, ys[xId], fVal, -1);
                });
            });

            // 3) sum prefixes, but only for this new feature

            right->hist_->prefixSumBinsLastFeature(-1);
        }

        left->usedFeatures_ = usedFeatures_;
        right->usedFeatures_ = usedFeatures_;
        left->usedFeaturesInOrder_ = usedFeaturesInOrder_;
        right->usedFeaturesInOrder_ = usedFeaturesInOrder_;

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

//ThreadPool GreedyLinearObliviousTreeLearner::buildThreadPool_;

ModelPtr GreedyLinearObliviousTreeLearnerV2::fit(const DataSet& ds, const Target& target) {
    auto tree = std::make_shared<LinearObliviousTreeV2>(grid_);

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

    DataSet curDs(ds.sampleMx({biasCol_}), target.targets());

    std::vector<int32_t> xIds(ds.samplesCount());
    std::iota(xIds.begin(), xIds.end(), 0);

    // build root

    auto root = std::make_shared<LinearObliviousTreeLeafV2>(bds, this->grid_, l2reg_, traceReg_, maxDepth_ + 1, 1, -1);
    root->usedFeatures_.insert(biasCol_);
    root->usedFeaturesInOrder_.push_back(biasCol_);
    root->hist_->build(curDs, xIds);
    root->hist_->prefixSumBins();

    leaves.emplace_back(std::move(root));

    for (unsigned int d = 0; d < maxDepth_; ++d) {

        // Update with new features

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        parallelFor<31>(0, grid_->nzFeaturesCount(), [&](int newF) {
            auto newOrigF = grid_->origFeatureIndex(newF);
            if (usedFeatures.count(newOrigF) != 0) return;
            Vec fColumn(ds.samplesCount());
            auto fColumn_ref = fColumn.arrayRef();
            ds.copyColumn(newOrigF, &fColumn);
            bds.visitFeature(newF, [&](int i, int8_t localBinId) {
                Vec x = curDs.sample(i);
                double fVal = fColumn_ref[i];
                unsigned int lId = leafId[i];
                leaves[lId]->updateBin(newF, localBinId, x, ys[i], fVal, 0);
            });
        });

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        parallelFor<31>(0, leaves.size(), [&](int lId) {
            auto& l = leaves[lId];
            l->hist_->prefixSumBinsLastFeature(0);
//            std::cout << "PRE SPLIT {" << std::endl;
//            l->printHists();
//            std::cout << "}" << std::endl;
        });

        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        std::cout << "Hists built in " << time_ms << " [ms], finding best split" << std::endl;

        // Find best split

        double bestSplitScore = 1e9;
        int32_t splitFId = -1;
        int32_t splitCond = -1;

        std::vector<std::vector<double>> splitScores;
        for (int64_t fId = 0; fId < grid_->nzFeaturesCount(); ++fId) {
            std::vector<double> fSplitScores;
            fSplitScores.resize(grid_->conditionsCount(fId), 0.0);
            splitScores.emplace_back(std::move(fSplitScores));
        }

        parallelFor<133>(0, grid_->nzFeaturesCount(), [&](int fId) {
            parallelFor<134>(0, grid_->conditionsCount(fId), [&](int cond) {
                for (auto& l : leaves) {
                   splitScores[fId][cond] += l->splitScore(fId, cond);
                }
//                std::cout << "fId: " << fId << ", cond: " << cond << ", split score: " << score << std::endl;
            });
        });

        for (int fId = 0; fId < grid_->nzFeatures().size(); ++fId) {
            for (int64_t cond = 0; cond < grid_->conditionsCount(fId); ++cond) {
                double sScore = splitScores[fId][cond];
                if (sScore < bestSplitScore) {
                    bestSplitScore = sScore;
                    splitFId = fId;
                    splitCond = cond;
                }
            }
        }

        std::cout << "Best split found, splitting" << std::endl;

        // Split

        int32_t splitOrigFId = grid_->origFeatureIndex(splitFId);
        Vec fColumn(ds.samplesCount());
        ds.copyColumn(splitOrigFId, &fColumn);


        std::vector<std::shared_ptr<LinearObliviousTreeLeafV2>> newLeaves;
        for (int i = 0; i < leaves.size() * 2; ++i) {
            newLeaves.emplace_back(std::shared_ptr<LinearObliviousTreeLeafV2>(nullptr));
        }

        parallelFor<31>(0, leaves.size(), [&](int lId) {
            auto& l = leaves[lId];
            auto splits = l->split(curDs, fColumn, leafId, splitFId, splitCond);
            newLeaves[splits.first->id_] = std::move(splits.first);
            newLeaves[splits.second->id_] = std::move(splits.second);
        });

        if (usedFeatures.count(splitOrigFId) == 0) {
            curDs.addColumn(fColumn);
            usedFeatures.insert(splitOrigFId);
        }

        std::cout << "Split done" << std::endl;

//        std::cout << "\n\n\n==================\n\n" << std::endl;

        leaves = std::move(newLeaves);

//        for (auto& l : leaves) {
//            std::cout << "POST SPLIT {" << std::endl;
//            l->printHists();
//            std::cout << "}" << std::endl;
//        }
    }

//    std::cout << "\n\n\n==================\n\n" << std::endl;


    parallelFor<31>(0, leaves.size(), [&](int lId) {
        auto& l = leaves[lId];
        l->fit(curDs);
//            std::cout << "PRE SPLIT {" << std::endl;
//            l->printInfo();
//            std::cout << "}" << std::endl;
    });

    tree->leaves_ = std::move(leaves);
    tree->usedFeatures_ = std::move(usedFeatures);

    return tree;
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
