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
