#include "greedy_linear_oblivious_trees.h"

#include <memory>
#include <set>

#include <core/vec_factory.h>
#include <core/matrix.h>

void Histogram::build(const DataSet& ds, const std::set<int>& usedFeatures,
        const std::vector<int32_t>& indices) {
    for (int i = 0, curF = 0; i < (int)bds_->totalBins(); ++i) {
        int nFeatures = usedFeatures.size();
        if (curF < bds_->gridPtr()->nzFeaturesCount() - 1
                && bds_->binOffsets().at(curF + 1) == i) {
            ++curF;
        }
        int curOrigF = bds_->gridPtr()->nzFeatures().at(curF).origFeatureId_;

        int vecSize = usedFeatures.size();
        if (usedFeatures.find(curOrigF) == usedFeatures.end()) {
            ++vecSize;
        }

        hist_XTX_.emplace_back(vecSize, vecSize);
        histLeft_XTX_.emplace_back(vecSize, vecSize);
        hist_XTy_.emplace_back(vecSize, 1);
        histLeft_XTy_.emplace_back(vecSize, 1);
        hist_cnt_.emplace_back(0);
        histLeft_cnt_.emplace_back(0);
    }

    auto target_arr_ref = ds.target().arrayRef();

    Detail::ArrayRef<const int32_t> indicesVecRef(indices.data(), indices.size());

    auto features = bds_->grid().nzFeatures();
    for (int f = 0; f < (int)features.size(); ++f) {
        auto newUsedFeatures = usedFeatures;
        newUsedFeatures.insert(features[f].origFeatureId_);
        bds_->visitFeature(features[f].featureId_, indicesVecRef, [&](int sampleId, int8_t localBinId) {
            int offset = bds_->binOffsets()[f];
            int binPos = offset + localBinId;

            auto x = ds.sample(sampleId, newUsedFeatures);
            auto X = Mx(x, x.size(), 1);
            auto XT = X.T();

            auto XTX = X * XT;
            hist_XTX_[binPos] += XTX;
            histLeft_XTX_[binPos] += XTX;

            auto XTy = XT * target_arr_ref[sampleId];
            hist_XTy_[binPos] += XTy;
            histLeft_XTy_[binPos] += XTy;

            hist_cnt_[binPos] += 1;
            histLeft_cnt_[binPos] += 1;
        }, false);

        int offset = bds_->binOffsets()[f];

        for (int localBinId = 1; localBinId <= bds_->gridPtr()->borders(f).size(); ++localBinId) {
            int binPos = offset + localBinId;
            histLeft_XTX_[binPos] += histLeft_XTX_[binPos - 1];
            histLeft_XTy_[binPos] += histLeft_XTy_[binPos - 1];
            histLeft_cnt_[binPos] += histLeft_cnt_[binPos - 1];
        }
    }
}

double Histogram::computeScore(Mx& XTX, Mx& XTy, uint32_t cnt) {
    if (cnt == 0) {
        return 0;
    }

    // Dealing with matrix singularity
    // Not sure what we should do in this case...
    // For now just return 0.
    //
    // TODO it's actually better to check that XTX is singular directly, this check might not be enough
    if (cnt < XTX.ydim()) {
        return 0;
    }

    Mx w = XTX.inverse();

    Mx c1(XTy.T() * w);
    c1 *= -2;
    assert(c1.xdim() == 1 && c1.ydim() == 1);

    Mx c2(w.T() * XTX * w);
    assert(c2.xdim() == 1 && c2.ydim() == 1);
    c2 += c1;

    return c2.get(0, 0);
}

std::pair<double, double> Histogram::splitScore(int fId, int condId) {
    uint32_t offset = bds_->binOffsets()[fId];
    uint32_t binPos = offset + condId;
    uint32_t lastPos = offset + bds_->gridPtr()->conditionsCount(fId);

    Mx left_XTX(histLeft_XTX_[binPos]);
    Mx right_XTX(Vec(histLeft_XTX_[lastPos].copy()), left_XTX.ydim(), left_XTX.xdim());
    right_XTX -= left_XTX;

    Mx left_XTy(histLeft_XTy_[binPos]);
    Mx right_XTy(Vec(histLeft_XTy_[lastPos].copy()), left_XTy.ydim(), left_XTy.xdim());
    right_XTy -= left_XTy;

    uint32_t left_cnt = histLeft_cnt_[binPos];
    uint32_t right_cnt = histLeft_cnt_[lastPos] - left_cnt;

    auto resLeft = computeScore(left_XTX, left_XTy, left_cnt);
    auto resRight = computeScore(right_XTX, right_XTy, right_cnt);

    return std::make_pair(resLeft, resRight);
}

Histogram operator-(const Histogram& lhs, const Histogram& rhs) {
    Histogram res(lhs.bds_);

    for (int32_t i = 0; i < lhs.bds_->totalBins(); ++i) {
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
    explicit LinearObliviousTreeLeaf(std::shared_ptr<GreedyLinearObliviousTree> tree)
            : tree_(std::move(tree)) {

    }

    void buildHist(const DataSet& ds) {
        if (hist_) {
            return;
        }

//        hist_ = std::make_unique<Histogram>(tree_->bds());
//        hist_->build(ds, usedFeatures_);
    }

    double score() {
        return score_;
    }

    double splitScore(const DataSet& ds, const Target& target, int fId, int condId) {
        buildHist(ds);
        return 0;

//        auto stats = hist_->splitStats(fId, condId);

//        return computeScore(ds, target, stats.first) + computeScore(ds, target, stats.second);
    }

    void fit(const DataSet& ds, const Target& target) {
        if (!hist_) {
            buildHist(ds);
        }

//        score_ = computeScore(ds, target, hist_->sumByFeature(0));
    }

    std::pair<std::shared_ptr<LinearObliviousTreeLeaf>, std::shared_ptr<LinearObliviousTreeLeaf>> split(const DataSet& ds, int32_t binIdx) {
        auto left = std::make_shared<LinearObliviousTreeLeaf>(tree_);
        auto right = std::make_shared<LinearObliviousTreeLeaf>(tree_);

        initChildren(ds, left, right, binIdx);

        return std::make_pair(left, right);
    }

private:
    double computeScore(const DataSet& ds, const Target& t) {
        auto dsX = ds.samplesMx();
        return t.value(dsX * (*coefs_));
    }

    double computeScore(const DataSet& ds, const Target& t, Mx stat) {
        Vec y = ds.target();
        Mx X = std::move(stat);
        Mx Y(y, y.size(), 1);
        Mx coefs = (X.T() * X).inverse() * X.T() * Y;
        coefs_ = std::make_unique<Mx>(coefs);

        return computeScore(ds, t);
    }

    void initChildren(const DataSet& ds, std::shared_ptr<LinearObliviousTreeLeaf>& left, std::shared_ptr<LinearObliviousTreeLeaf>& right, int32_t binIdx) {
        int32_t featureId = tree_->gridPtr()->binFeature(binIdx).featureId_;
        double border = tree_->gridPtr()->condition(binIdx);

        for (auto xId : xIds_) {
            if (ds.sample(xId)(featureId) <= border) {
                left->xIds_.push_back(xId);
            } else {
                right->xIds_.push_back(xId);
            }
        }

        left->usedFeatures_ = this->usedFeatures_;
        left->usedFeatures_.insert(featureId);
        right->usedFeatures_ = this->usedFeatures_;
        right->usedFeatures_.insert(featureId);
    }

private:
    std::shared_ptr<GreedyLinearObliviousTree> tree_;
//    int splitBinIdx_ = -1;
    std::vector<uint64_t> xIds_;
//    std::vector<int32_t> splits_;
    std::set<int32_t> usedFeatures_;
    std::unique_ptr<Mx> coefs_;
    double score_ = 1e9;

    std::unique_ptr<Histogram> hist_;
};

//ModelPtr GreedyLinearObliviousTree::fit(const DataSet& ds, const Target& target) {
//    std::vector<std::shared_ptr<LinearObliviousTreeLeaf>> new_leaves;
////    for (int d = 0; d < maxDepth_; ++d) {
////        for (auto& l : leaves_) {
////            l->buildHist(ds);
////        }
////
////        for ()
////    }
//
//    return std::make_shared<GreedyLinearObliviousTree>(*this);
//}
