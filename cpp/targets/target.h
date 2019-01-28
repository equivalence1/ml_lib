#pragma once

#include <data/dataset.h>
#include <core/vec.h>
#include <core/func.h>
#include <util/array_ref.h>
#include <core/buffer.h>

class Target : public virtual FuncC1 {
public:

    virtual const DataSet& owner() const = 0;
};

class PointwiseTarget : public Object {
public:
    virtual void der(const Vec& point, const Buffer<uint32_t>& indices, Vec to) = 0;
};

class PointwiseC2Target : public PointwiseTarget {

    virtual void derAndDer2(const Vec& point,
                            const Buffer<uint32_t>& indices,
                            Vec derTo,
                            Vec der2To) = 0;

    virtual void der2(const Vec& point, const Buffer<uint32_t>& indices, Vec to) = 0;

};

template <class Impl>
class Stub<Target, Impl> : public virtual Target, public Stub<FuncC1, Impl> {
public:

    std::unique_ptr<Trans> gradient() const override {
        return std::unique_ptr<Trans>(new typename Impl::Der(*static_cast<const Impl*>(this)));
    }


    Stub(const DataSet& ds)
    : Stub<FuncC1, Impl>(ds.samplesCount())
    , ds_(ds) {}


    const DataSet& owner() const override {
        return ds_;
    }
private:
    const DataSet& ds_;

};


class EmpiricalTargetFactory : public Object {
public:
    virtual SharedPtr<Target> create(const DataSet& ds,
                                     const Target& target,
                                     const Mx& startPoint) = 0;
};





