#include "greedy_linear_oblivious_trees.h"

#include <memory>
#include <set>
#include <stdexcept>

#include <core/vec_factory.h>
#include <core/matrix.h>

void Histogram::build(const DataSet& ds, const std::set<int>& usedFeatures,
        const std::vector<int32_t>& indices) {
    const auto& bds = cachedBinarize(ds, grid_);

    hist_XTX_ = std::vector<Mx>();
    histLeft_XTX_ = std::vector<Mx>();
    hist_XTy_ = std::vector<Mx>();
    histLeft_XTy_ = std::vector<Mx>();
    hist_cnt_ = std::vector<uint32_t>();
    histLeft_cnt_ = std::vector<uint32_t>();

    if (!usedFeatures.empty()) {
        usedFeature_ = *usedFeatures.begin();

        auto features = grid_->nzFeatures();
        for (int f = 0; f < (int)features.size(); ++f) {
            if (features.at(f).origFeatureId_ == usedFeature_) {
                usedFeature_ = f;
                break;
            }
        }
    }

    for (int i = 0, curF = 0; i < (int)bds.totalBins(); ++i) {
        if (curF < grid_->nzFeaturesCount() - 1
                && bds.binOffsets().at(curF + 1) == i) {
            ++curF;
        }
        int curOrigF = grid_->nzFeatures().at(curF).origFeatureId_;

        int vecSize = usedFeatures.size();
        if (usedFeatures.find(curOrigF) == usedFeatures.end()) {
            ++vecSize;
        }

        // bias term
        vecSize += 1;

        hist_XTX_.emplace_back(vecSize, vecSize);
        histLeft_XTX_.emplace_back(vecSize, vecSize);
        hist_XTy_.emplace_back(vecSize, 1);
        histLeft_XTy_.emplace_back(vecSize, 1);
        hist_cnt_.emplace_back(0);
        histLeft_cnt_.emplace_back(0);
    }

    auto y = ds.target().arrayRef();

    Detail::ArrayRef<const int32_t> indicesVecRef(indices.data(), indices.size());

    auto features = grid_->nzFeatures();
    for (int f = 0; f < (int)features.size(); ++f) {
        auto newUsedFeatures = usedFeatures;
        newUsedFeatures.insert(features[f].origFeatureId_);
        std::cout << "newUsedFeatures.size(): " << newUsedFeatures.size() << std::endl;
        bds.visitFeature(features[f].featureId_, indicesVecRef, [&](int idxId, int8_t localBinId) {
            int offset = bds.binOffsets()[f];
            int binPos = offset + localBinId;

            int32_t sampleId = indicesVecRef.at(idxId);

            auto x = ds.sample(sampleId, newUsedFeatures);
            auto xb = x.append(1);
            auto X = Mx(xb, xb.size(), 1);
            auto XT = X.T();

            std::cout << "fId: " << f << ", bin: " << int(localBinId) << ", adding x " << X <<
                     ", target " << std::setprecision(3) << y[sampleId] << std::endl;

            auto XTX = X * XT;
            hist_XTX_[binPos] += XTX;
            histLeft_XTX_[binPos] += XTX;

            auto XTy = XT * y[sampleId];
            hist_XTy_[binPos] += XTy;
            histLeft_XTy_[binPos] += XTy;

            hist_cnt_[binPos] += 1;
            histLeft_cnt_[binPos] += 1;
        }, false);

        int offset = bds.binOffsets()[f];

        for (int localBinId = 1; localBinId <= grid_->conditionsCount(f); ++localBinId) {
            int binPos = offset + localBinId;
            histLeft_XTX_[binPos] += histLeft_XTX_[binPos - 1];
            histLeft_XTy_[binPos] += histLeft_XTy_[binPos - 1];
            histLeft_cnt_[binPos] += histLeft_cnt_[binPos - 1];
        }
    }

    std::cout << "built" << std::endl;
}

std::shared_ptr<Mx> Histogram::getW() {
    if (usedFeature_ == -1) {
        throw std::runtime_error("No features are used");
    }

    uint32_t offset = grid_->binOffsets().at(usedFeature_);
    uint32_t lastPos = offset + grid_->conditionsCount(usedFeature_);

    Mx& XTX = histLeft_XTX_[lastPos];
    Mx& XTy = histLeft_XTy_[lastPos];

    // -1 is because of bias
    if (histLeft_cnt_[lastPos] <= XTX.ydim() - 1) {
        return std::make_shared<Mx>(XTX.ydim(), XTX.xdim());
    }

    try {
        XTX.inverse();
    } catch (...) {
        std::cout << "No inverse" << std::endl;
        return std::shared_ptr<Mx>(nullptr);
    }

    return std::make_shared<Mx>(XTX.inverse() * XTy);
}

double Histogram::computeScore(Mx& XTX, Mx& XTy, uint32_t cnt) {
    if (cnt == 0) {
        return 0;
    }

    // Dealing with matrix singularity
    // Not sure what we should do in this case...
    // For now just return 0.
    try {
        XTX.inverse();
    } catch (...) {
        std::cout << "No inverse" << std::endl;
        return 0;
    }

    Mx w = XTX.inverse() * XTy;
    std::cout << "w is " << w << std::endl;

    Mx c1(XTy.T() * w);
    c1 *= -2;
    assert(c1.xdim() == 1 && c1.ydim() == 1);

    Mx c2(w.T() * XTX * w);
    assert(c2.xdim() == 1 && c2.ydim() == 1);
    c2 += c1;

    return c2.get(0, 0);
}

std::pair<double, double> Histogram::splitScore(int fId, int condId) {
    uint32_t offset = grid_->binOffsets()[fId];
    uint32_t binPos = offset + condId;
    uint32_t lastPos = offset + grid_->conditionsCount(fId);

    Mx left_XTX(histLeft_XTX_[binPos]);
    Mx right_XTX(Vec(histLeft_XTX_[lastPos].copy()), left_XTX.ydim(), left_XTX.xdim());
    right_XTX -= left_XTX;

    Mx left_XTy(histLeft_XTy_[binPos]);
    Mx right_XTy(Vec(histLeft_XTy_[lastPos].copy()), left_XTy.ydim(), left_XTy.xdim());
    right_XTy -= left_XTy;

    uint32_t left_cnt = histLeft_cnt_[binPos];
    uint32_t right_cnt = histLeft_cnt_[lastPos] - left_cnt;

    std::cout << "split fId: " << fId << ", cond: " << condId << " ";

    auto resLeft = computeScore(left_XTX, left_XTy, left_cnt);
    auto resRight = computeScore(right_XTX, right_XTy, right_cnt);

    return std::make_pair(resLeft, resRight);
}

Histogram operator-(const Histogram& lhs, const Histogram& rhs) {
    Histogram res(lhs.grid_);

    for (int32_t i = 0; i < lhs.grid_->totalBins(); ++i) {
        res.hist_XTX_.emplace_back(lhs.hist_XTX_[i] - rhs.hist_XTX_[i]);
        res.histLeft_XTX_.emplace_back(lhs.histLeft_XTX_[i] - rhs.histLeft_XTX_[i]);

        res.hist_XTy_.emplace_back(lhs.hist_XTy_[i] - rhs.hist_XTy_[i]);
        res.histLeft_XTy_.emplace_back(lhs.histLeft_XTy_[i] - rhs.histLeft_XTy_[i]);

        res.hist_cnt_.emplace_back(lhs.hist_cnt_[i] - rhs.hist_cnt_[i]);
        res.hist_cnt_.emplace_back(lhs.histLeft_cnt_[i] - rhs.histLeft_cnt_[i]);
    }

    return res;
}


class LinearObliviousTreeLeaf : std::enable_shared_from_this<LinearObliviousTreeLeaf> {
public:
    explicit LinearObliviousTreeLeaf(GridPtr grid)
            : grid_(std::move(grid)) {

    }

    void buildHist(const DataSet& ds) {
        if (hist_) {
            return;
        }

        std::cout << "building hist with indices: ";
        for (auto xId : xIds_) {
            std::cout << xId << " ";
        }
        std::cout << std::endl;

        hist_ = std::make_unique<Histogram>(grid_);
        hist_->build(ds, usedFeatures_, xIds_);
    }

    double splitScore(const DataSet& ds, const Target& target, int fId, int condId) {
        buildHist(ds);
        auto sScore = hist_->splitScore(fId, condId);
        return sScore.first + sScore.second;
    }

    void fit(const DataSet& ds, const Target& target) {
        if (w_) {
            return;
        }
        if (!hist_) {
            buildHist(ds);
        }
        w_ = hist_->getW();
    }

    double value(const Vec& x) {
        if (!w_) {
            throw std::runtime_error("Not fitted");
        }

        std::vector<double> tmp;
        for (auto f : usedFeatures_) {
            tmp.push_back(x(f));
        }
        auto xVec = VecFactory::fromVector(tmp).append(1);
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

    std::pair<std::shared_ptr<LinearObliviousTreeLeaf>, std::shared_ptr<LinearObliviousTreeLeaf>> split(const DataSet& ds,
            int32_t fId, int32_t condId) {
        auto left = std::make_shared<LinearObliviousTreeLeaf>(grid_);
        auto right = std::make_shared<LinearObliviousTreeLeaf>(grid_);

        initChildren(ds, left, right, fId, condId);

        return std::make_pair(left, right);
    }

private:
    void initChildren(const DataSet& ds, std::shared_ptr<LinearObliviousTreeLeaf>& left,
            std::shared_ptr<LinearObliviousTreeLeaf>& right,
            int32_t fId, int32_t condId) {
        int32_t origFeatureId = grid_->nzFeatures().at(fId).origFeatureId_;
        double border = grid_->borders(fId).at(condId);

        std::cout << "Splitting on feature " << origFeatureId << std::endl;

        for (auto xId : xIds_) {
            if (ds.sample(xId)(origFeatureId) <= border) {
                left->xIds_.push_back(xId);
            } else {
                right->xIds_.push_back(xId);
            }
        }

        left->usedFeatures_ = this->usedFeatures_;
        left->usedFeatures_.insert(origFeatureId);
        right->usedFeatures_ = this->usedFeatures_;
        right->usedFeatures_.insert(origFeatureId);

        left->splits_ = this->splits_;
        left->splits_.emplace_back(std::make_tuple(fId, condId, true));
        right->splits_ = this->splits_;
        right->splits_.emplace_back(std::make_tuple(fId, condId, false));
    }

private:
    friend class GreedyLinearObliviousTree;

    GridPtr grid_;
    std::vector<int32_t> xIds_;
    std::set<int32_t> usedFeatures_;
    std::shared_ptr<Mx> w_;
    std::vector<std::tuple<int32_t, int32_t, bool>> splits_;

    std::unique_ptr<Histogram> hist_;
};

ModelPtr GreedyLinearObliviousTree::fit(const DataSet& ds, const Target& target) {
    auto root = std::make_shared<LinearObliviousTreeLeaf>(this->grid_);
    root->xIds_.resize(ds.samplesCount());
    std::iota(root->xIds_.begin(), root->xIds_.end(), 0);

    leaves_.push_back(std::move(root));

    for (int d = 0; d < maxDepth_; ++d) {
        double bestSplitScore = 1e9;
        int32_t splitFId = -1;
        int32_t splitCond = -1;

        for (int fId = 0; fId < gridPtr()->nzFeatures().size(); ++fId) {
            for (int cond = 0; cond < gridPtr()->conditionsCount(fId); ++cond) {
                double splitScore = 0;
                for (auto& l : leaves_) {
                    splitScore += l->splitScore(ds, target, fId, cond);
                }

                if (splitScore < bestSplitScore) {
                    bestSplitScore = splitScore;
                    splitFId = fId;
                    splitCond = cond;
                }
            }
        }

        std::vector<std::shared_ptr<LinearObliviousTreeLeaf>> new_leaves;
        for (auto& l : leaves_) {
            auto nLeaves = l->split(ds, splitFId, splitCond);
            new_leaves.emplace_back(std::move(nLeaves.first));
            new_leaves.emplace_back(std::move(nLeaves.second));
        }

        leaves_ = std::move(new_leaves);
    }

    for (auto& l : leaves_) {
        l->fit(ds, target);
    }

    return std::make_shared<GreedyLinearObliviousTree>(*this);
}

void GreedyLinearObliviousTree::appendTo(const Vec &x, Vec to) const {
    throw std::runtime_error("Unimplemented");
}

double GreedyLinearObliviousTree::value(const Vec &x) {
    for (auto& l : leaves_) {
        if (l->isInRegion(x)) {
            return scale_ * l->value(x);
        }
    }

    throw std::runtime_error("given x does not belong to any region O_o");
}

void GreedyLinearObliviousTree::grad(const Vec &x, Vec to) {
    throw std::runtime_error("Unimplemented");
}
