#pragma once
#include <core/trans.h>
#include <data/dataset.h>
#include <vec_tools/fill.h>

//todo: in model
enum class ApplyType {
    Append,
    Set
};

class Model : public virtual Trans {
public:

    void apply(const DataSet& ds, Mx to) const {
        applyToDs(ds, to);
    }

    void append(const DataSet& ds, Mx to) const {
        appendToDs(ds, to);
    }

    operator std::unique_ptr<Model>() const {
        return cloneModelUnique(1.0);
    }

    operator std::shared_ptr<Model>() const {
        return cloneModelShared(1.0);
    }

    Vec trans(const Vec& x, Vec to) const override {
        VecTools::fill(0, to);
        appendTo(x, to);
        return to;
    }

    //todo: should be scaled model, but i'm lazy :)
    std::shared_ptr<Model> scale(double alpha) const {
        return cloneModelShared(alpha);
    }

    virtual void appendTo(const Vec& x, Vec to) const = 0;
    virtual double value(const Vec& x) { return 0; }
    virtual void grad(const Vec& x, Vec to) {}

protected:

    virtual void applyToDs(const DataSet& ds, Mx to) const {
        assert(to.ydim() == ds.samplesCount());
        for (int64_t i = 0; i < ds.samplesCount(); ++i) {
            trans(ds.sample(i), to.row(i));
        }
    }


    virtual void appendToDs(const DataSet& ds, Mx to) const {
        assert(to.ydim() == ds.samplesCount());
        for (int64_t i = 0; i < ds.samplesCount(); ++i) {
            appendTo(ds.sample(i), to.row(i));
        }
    }


protected:
    virtual std::unique_ptr<Model> cloneModelUnique(double scale) const = 0;
    virtual std::shared_ptr<Model> cloneModelShared(double scale) const = 0;
};


template <class Impl>
class Stub<Model, Impl> : public virtual Model, public Stub<Trans, Impl> {
public:
    Stub(int64_t xdim, int64_t ydim)
    : Stub<Trans, Impl>(xdim, ydim) {

    }

protected:
    std::unique_ptr<Model> cloneModelUnique(double scale) const override {
        return std::unique_ptr<Model>(new Impl(*static_cast<const Impl*>(this), scale));
    }

    std::shared_ptr<Model> cloneModelShared(double scale) const override {
        return std::static_pointer_cast<Model>(std::make_shared<Impl>(*static_cast<const Impl*>(this), scale));
    }
};


using ModelPtr = std::shared_ptr<Model>;
