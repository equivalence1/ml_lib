#include "oblivious_tree.h"


void ObliviousTree::applyBinarizedRow(const Buffer<uint8_t>& x, Vec to) const {
    assert(x.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.dim() == 1);
    auto bytes = x.arrayRef();

    int32_t bin = 0;
    for (int64_t f = 0; f < splits_.size(); ++f) {
        if (bytes[splits_[f].featureId_] > splits_[f].conditionId_) {
            bin |= 1 << f;
        }
    }

    to.set(0, static_cast<float>(leaves_.get(bin)));
}

void ObliviousTree::applyToBds(const BinarizedDataSet& ds, Mx to, ApplyType type) const {
    assert(to.ydim() == ds.samplesCount());
    auto bins = Buffer<uint32_t>::create(ds.samplesCount());
    bins.fill(0);
    auto binsArray = bins.arrayRef();

    for (int64_t i = 0; i < splits_.size(); ++i) {
        auto binFeature = splits_[i];
        //TODO:: this is map operation (but for non-trivial accessor)
        ds.visitFeature(binFeature.featureId_, [&](const int64_t lineIdx, int32_t bin) {
            if (bin > binFeature.conditionId_) {
                binsArray[lineIdx] |= (1 << i);
            }
        }, true);
    }


    ArrayRef<float> dstArray = static_cast<Vec>(to).arrayRef();
    ConstArrayRef<float> leavesRef = leaves_.arrayRef();

    if (type == ApplyType::Set) {
        //TODO(noxoomo): this is gather primitive
        parallelFor(0, binsArray.size(), [&](int64_t i) {
            dstArray[i] = leavesRef[binsArray[i]];
        });
    } else {
        parallelFor(0, binsArray.size(), [&](int64_t i) {
            dstArray[i] += leavesRef[binsArray[i]];
        });
    }
}


void ObliviousTree::appendTo(const Vec& x, Vec to) const {
    assert(x.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.device().deviceType() == ComputeDeviceType::Cpu);
    assert(to.dim() == 1);


    int32_t bin = 0;
    for (int64_t f = 0; f < splits_.size(); ++f) {
        const auto binFeature = splits_[f];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = x.get(grid_->origFeatureIndex(binFeature.featureId_));
        if (val > border) {
            bin |= 1 << f;
        }
    }

    to += leaves_.get(bin);
}

double ObliviousTree::value(const Vec& x) {

    std::vector<double> vec(weights_.dim());
    std::vector<double> probs(splits_.size());
    auto weightsPtr = weights_.arrayRef();
    for (int64_t i = 0; i < probs.size(); ++i) {
        const auto binFeature = splits_[i];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = x.get(grid_->origFeatureIndex(binFeature.featureId_));
        probs[i] = 1./(1. + exp(-(val - border)));
    }
    for (int b = 0; b < vec.size(); ++b) {
        double value = 0;
        int64_t bitsB = bits(b);
        for (int a = 0; a < vec.size(); ++a) {
            int64_t bitsA = bits(a);
            if (bits(a & b) >= bitsA) {
                value += (((bitsA + bitsB) & 1) > 0 ? -1. : 1.) * weightsPtr[a];
            }
        }
        for (int f = 0; f < probs.size(); ++f) {
            if ((b >> f & 1) != 0) {
                value *= probs[probs.size() - f - 1];
            }
        }
        vec[b] = value;
    }
    double res = 0;
    for (int64_t i = 0; i < vec.size(); ++i) {
        res += vec[i] * weights_.get(i);
    }
    return res;
}

void ObliviousTree::grad(const Vec& x, Vec to) {
    std::vector<int64_t> masks(x.dim());
    for (int i = 0; i < splits_.size(); ++i) {
        const auto binFeature = splits_[i];
        masks[grid_->origFeatureIndex(binFeature.featureId_)] += (1 << (splits_.size() - i - 1));
    }

    std::vector<double> vec(weights_.dim());
    std::vector<double> probs(splits_.size());
    for (int64_t i = 0; i < probs.size(); i++) {
        const auto binFeature = splits_[i];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = x.get(grid_->origFeatureIndex(binFeature.featureId_));
        probs[i] = 1./(1. + exp(-(val - border)));
    }
    for (int i = 0; i < to.dim(); ++i) {

        for (int b = 0; b < vec.size(); ++b) {
            if (masks[i] & b == 0) {
                vec[b] = 0.;
                continue;
            }
            double value = 0;
            int64_t bitsB = bits(b);
            for (int a = 0; a < vec.size(); ++a) {
                int64_t bitsA = bits(a);
                if (bits(a & b) >= bitsA)
                    value += (((bitsA + bitsB) & 1) > 0 ? -1 : 1) * weights_.get(a);
            }
            double diffCf = 0;
            for (int f = 0; f < probs.size(); f++) {
                if ((b >> f & 1) != 0) {
                    value *= probs[probs.size() - f - 1];
                    if ((masks[i] >> f & 1) != 0) {
                        diffCf += 1 - probs[probs.size() - f - 1];
                    }
                }
            }
            value *= diffCf;
            vec[b] = value;
        }
        double res = 0;
        for (int64_t k = 0; k < vec.size(); ++k) {
            res += vec[k] * weights_.get(k);
        }
        to.set(i, res);
    }

}

//TODO: special instrict for intel
int64_t ObliviousTree::bits(int64_t x) {
    int64_t result = 0;
    while (x != 0) {
        result += (x & 1);
        x >>= 1;
    }
    return result;
}
