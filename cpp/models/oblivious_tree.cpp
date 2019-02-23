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

    std::vector<double> vec(leaves_.dim());
    std::vector<double> probs(splits_.size());
    auto leavesPtr = leaves_.arrayRef();
    for (int64_t i = 0; i < probs.size(); ++i) {
        const auto binFeature = splits_[i];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = x.get(grid_->origFeatureIndex(binFeature.featureId_));
        probs[i] = 1./(1. + exp(-(val - border)));
    }
    for (uint b = 0; b < vec.size(); ++b) {
        double value = 0;
        uint bitsB = bits(b);
        for (uint a = 0; a < vec.size(); ++a) {
            uint bitsA = bits(a);
            if (bits(a & b) >= bitsA) {
                value += (((bitsA + bitsB) & 1) > 0 ? -1. : 1.) * leavesPtr[a];
            }
        }
        for (int f = 0; f < probs.size(); ++f) {
            if (((b >> f) & 1) != 0) {
                value *= probs[probs.size() - f - 1];
            }
        }
        vec[b] = value;
    }
    double res = 0;
    for (int64_t i = 0; i < vec.size(); ++i) {
        res += vec[i] * leavesPtr[i];
    }
    return res;
}

void ObliviousTree::grad(const Vec& x, Vec to) {
    std::vector<int64_t> masks(x.dim());
    for (int i = 0; i < splits_.size(); ++i) {
        const auto binFeature = splits_[i];
        masks[grid_->origFeatureIndex(binFeature.featureId_)] += (1 << (splits_.size() - i - 1));
    }

    std::vector<double> vec(leaves_.dim());
    std::vector<double> probs(splits_.size());
    auto leavesPtr = leaves_.arrayRef();
    for (int64_t i = 0; i < probs.size(); i++) {
        const auto binFeature = splits_[i];
        const auto border = grid_->condition(binFeature.featureId_, binFeature.conditionId_);
        const auto val = x.get(grid_->origFeatureIndex(binFeature.featureId_));
        probs[i] = 1./(1. + exp(-(val - border)));
    }
    for (int i = 0; i < to.dim(); ++i) {

        for (uint b = 0; b < vec.size(); ++b) {
            if ((masks[i] & b) == 0) {
                vec[b] = 0.;
                continue;
            }
            double value = 0;
            uint bitsB = bits(b);
            for (uint a = 0; a < vec.size(); ++a) {
                uint bitsA = bits(a);
                if (bits(a & b) >= bitsA)
                    value += (((bitsA + bitsB) & 1) > 0 ? -1 : 1) * leavesPtr[a];
            }
            double diffCf = 0;
            for (int f = 0; f < probs.size(); f++) {
                if (((b >> f) & 1) != 0) {
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
            res += vec[k] * leavesPtr[k];
        }
        to.set(i, res);
    }

}

uint ObliviousTree::bits(uint i) {
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
//    return __builtin_popcount(x);
//    uint result = 0;
//    while (x != 0) {
//        result += (x & 1);
//        x >>= 1;
//    }
//    return result;
}