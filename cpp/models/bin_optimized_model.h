#pragma once

#include "model.h"
#include <data/dataset.h>
#include <data/binarized_dataset.h>

class BinOptimizedModel : public virtual Model {
public:
    using Model::apply;

    void apply(const BinarizedDataSet& ds, Mx to) const {
        applyToBds(ds, to);
    }

     operator std::unique_ptr<BinOptimizedModel>() const {
        return cloneBinModelUnique();
     }

    operator std::shared_ptr<BinOptimizedModel>() const {
        return cloneBinModelShared();
    }

protected:
    virtual void applyToBds(const BinarizedDataSet& ds, Mx to) const = 0;

protected:
    virtual std::unique_ptr<BinOptimizedModel> cloneBinModelUnique() const  = 0;

    virtual std::shared_ptr<BinOptimizedModel> cloneBinModelShared() const  = 0;

    virtual std::unique_ptr<Model> cloneModelUnique() const {
        return std::unique_ptr<BinOptimizedModel>(cloneBinModelUnique().release());
    }

    virtual std::shared_ptr<Model> cloneModelShared() const {
        return std::static_pointer_cast<Model>(cloneBinModelShared());
    }
};



template <class Impl>
class BinOptimizedModelStub : public BinOptimizedModel, public TransStub<Impl> {
public:

    BinOptimizedModelStub(int64_t xdim, int64_t ydim)
    : TransStub<Impl>(xdim, ydim) {

    }

    BinOptimizedModelStub(const Impl& other)
    : TransStub<Impl>(other) {

    }

    Vec trans(const Vec& x, Vec to) const override {
        Buffer<uint8_t> bins = Buffer<uint8_t>::create(x.dim());
        static_cast<const Impl*>(this)->grid().binarize(x, bins);
        static_cast<const Impl*>(this)->applyBinarizedRow(bins, to);
        return to;
    }

protected:
    std::unique_ptr<BinOptimizedModel> cloneBinModelUnique() const override {
        return std::unique_ptr<BinOptimizedModel>(new Impl(*static_cast<const Impl*>(this)));
    }

    std::shared_ptr<BinOptimizedModel> cloneBinModelShared() const override {
        return std::static_pointer_cast<BinOptimizedModel>(std::make_shared<Impl>(*static_cast<const Impl*>(this)));
    }
};
