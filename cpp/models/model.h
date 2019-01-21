#pragma once
#include <core/trans.h>
#include <data/dataset.h>

class Model : public virtual Trans {
public:

    void apply(const DataSet& ds, Mx to) const {
        applyToDs(ds, to);
    }

    virtual void append(const DataSet& ds, Mx to) const {
        appendToDs(ds, to);
    }

    operator std::unique_ptr<Model>() const {
        return cloneModelUnique();
    }

    operator std::shared_ptr<Model>() const {
        return cloneModelShared();
    }

protected:

    virtual void applyToDs(const DataSet& ds, Mx to) const {
        assert(to.ydim() == ds.samplesCount());
        for (int64_t i = 0; i < ds.samplesCount(); ++i) {
            trans(ds.sample(i), to.row(i));
        }
    }


    virtual void appendToDs(const DataSet& ds, Mx to) const {
        Vec tmp(ydim());
        assert(to.ydim() == ds.samplesCount());

        for (int64_t i = 0; i < ds.samplesCount(); ++i) {
            trans(ds.sample(i), to.row(i));
        }
    }

protected:
    virtual std::unique_ptr<Model> cloneModelUnique() const = 0;
    virtual std::shared_ptr<Model> cloneModelShared() const = 0;
};


template <class Impl>
class ModelStub : public virtual Model, public TransStub<Impl> {
public:
    ModelStub(int64_t xdim, int64_t ydim)
    : TransStub<Impl>(xdim, ydim) {

    }

protected:
    std::unique_ptr<Model> cloneModelUnique() const override {
        return std::unique_ptr<Model>(new Impl(*static_cast<const Impl*>(this)));
    }

    std::shared_ptr<Model> cloneModelShared() const override {
        return std::static_pointer_cast<Model>(std::make_shared<Impl>(*static_cast<const Impl*>(this)));
    }
};


using ModelPtr = std::unique_ptr<Model>;
