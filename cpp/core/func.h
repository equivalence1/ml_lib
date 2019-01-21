#pragma once

#include "trans.h"

#include "object.h"
#include "batch.h"
#include "vec.h"

#include <memory>
#include <optional>
#include <functional>

using DoubleRef = double&;
using ConstDoubleRef = const double&;

//TODO(noxoomo): batch trans
class Func : public virtual Trans {
public:
    virtual int64_t dim() const = 0;

    virtual int64_t ydim() const final {
        return 1;
    };

    virtual double value(const Vec& x) const {
        Vec val(1);
        trans(x, val);
        return val.get(0);
    }

    operator std::unique_ptr<Func>() const {
        return cloneUniqueFunc();
    }

    operator std::shared_ptr<Func>() const {
        return cloneSharedFunc();
    }

    //    virtual Batch<Vec> trans(Batch<ConstVec> x, Batch<Vec> to) const = 0;

protected:

    virtual std::unique_ptr<Func> cloneUniqueFunc() const = 0;

    virtual std::shared_ptr<Func> cloneSharedFunc() const = 0;

    virtual std::unique_ptr<Trans> cloneUnique() const override {
        return std::unique_ptr<Trans>(cloneUniqueFunc().release());
    }

    virtual std::shared_ptr<Trans> cloneShared() const override {
        return std::dynamic_pointer_cast<Trans>(cloneSharedFunc());
    }

};

class FuncC1 : public virtual Func, public virtual TransC1 {
public:

    virtual Vec gradientTo(const Vec& x, Vec to) const = 0;

    Mx gradientTo(const Vec& x, Mx to) const final {
        return Mx(gradientTo(x, static_cast<Vec>(to)), x.dim(), 1);
    }

    Vec gradientRowTo(const Vec& x, Vec to, int64_t index) const final {
        assert(index == 0);
        return gradientTo(x, to);
    }

    operator std::unique_ptr<FuncC1>() const {
        return cloneUniqueFuncC1();
    }

    operator std::shared_ptr<FuncC1>() const {
        return cloneSharedFuncC1();
    }
protected:
    virtual std::unique_ptr<FuncC1> cloneUniqueFuncC1() const = 0;

    virtual std::shared_ptr<FuncC1> cloneSharedFuncC1() const = 0;

    virtual std::unique_ptr<TransC1> cloneUniqueC1() const override {
        return std::unique_ptr<TransC1>(cloneUniqueFuncC1().release());
    }

    virtual std::shared_ptr<TransC1> cloneSharedC1() const override {
        return std::dynamic_pointer_cast<TransC1>(cloneSharedFuncC1());
    }

};


template <class Impl>
class FuncStub : public virtual Func {
public:
    FuncStub(int64_t dim)
        : dim_(dim) {

    }

    int64_t dim() const final {
        return dim_;
    }

    int64_t xdim() const final {
        return dim_;
    }


    double value(const Vec& x) const {
        double result = 0;
        static_cast<const Impl*>(this)->valueTo(x, result);
        return result;
    }

    Vec trans(const Vec& x, Vec to) const final {
        static_cast<const Impl*>(this)->trans(x, to);
        return to;
    }

private:

    virtual std::unique_ptr<Func> cloneUniqueFunc() const {
        return std::unique_ptr<Func>(new Impl(*static_cast<const Impl*>(this)));
    }

    virtual std::shared_ptr<Func> cloneSharedFunc() const {
        return std::static_pointer_cast<Func>(std::make_shared<Impl>(*static_cast<const Impl*>(this)));
    }

private:
    int64_t dim_;
};

template <class Impl>
class FuncC1Stub : public FuncC1, public FuncStub<Impl> {
public:
    FuncC1Stub(int64_t dim)
        : FuncStub<Impl>(dim) {

    }

    Vec gradientTo(const Vec& x, Vec to) const {
        return static_cast<const Impl*>(this)->gradient()->trans(x, to);
    }

protected:
    virtual std::unique_ptr<FuncC1> cloneUniqueFuncC1() const final {
        return std::unique_ptr<FuncC1>(new Impl(*static_cast<const Impl*>(this)));
    }

    virtual std::shared_ptr<FuncC1> cloneSharedFuncC1() const final {
        return std::static_pointer_cast<FuncC1>(std::make_shared<Impl>(*static_cast<const Impl*>(this)));
    }
};


