#pragma once

#include "object.h"
#include "batch.h"
#include "vec.h"
#include "matrix.h"

#include <memory>
#include <optional>
#include <functional>

class Trans : public Object {
public:
    virtual int64_t xdim() const = 0;

    virtual int64_t ydim() const = 0;

    virtual Vec trans(const Vec& x, Vec to) const = 0;

    operator std::unique_ptr<Trans>() const {
        return cloneUnique();
    }

    operator std::shared_ptr<Trans>() const {
        return cloneShared();
    }
protected:

    virtual std::unique_ptr<Trans> cloneUnique() const = 0;

    virtual std::shared_ptr<Trans> cloneShared() const = 0;

};

class TransC1 : public virtual Trans {
public:
    virtual Mx gradientTo(const Vec& x, Mx to) const = 0;

    virtual Vec gradientRowTo(const Vec& x, Vec to, int64_t index) const = 0;

    virtual std::unique_ptr<Trans> gradient() const = 0;

    operator std::unique_ptr<TransC1>() const {
        return cloneUniqueC1();
    }

    operator std::shared_ptr<TransC1>() const {
        return cloneSharedC1();
    }

protected:

    virtual std::unique_ptr<TransC1> cloneUniqueC1() const = 0;

    virtual std::shared_ptr<TransC1> cloneSharedC1() const = 0;


};


template <class Impl>
class Stub<Trans, Impl> : public virtual Trans {
public:

    Stub(int64_t xdim, int64_t ydim)
        : xdim_(xdim)
          , ydim_(ydim) {

    }

    Stub(const Stub& other)
    : xdim_(other.xdim_)
    , ydim_(other.ydim_) {

    }

    int64_t xdim() const final {
        return xdim_;
    }

    int64_t ydim() const final {
        return ydim_;
    }
protected:
    std::unique_ptr<Trans> cloneUnique() const final {
        return std::unique_ptr<Trans>(new Impl(*static_cast<const Impl*>(this)));
    }

    std::shared_ptr<Trans> cloneShared() const final {
        return std::make_shared<Impl>(*static_cast<const Impl*>(this));
    }

private:
    int64_t xdim_;
    int64_t ydim_;
};

template <class Impl>
class MapStub : public Stub<Trans, Impl> {
public:
    MapStub(int64_t dim)
        : Stub<Trans, Impl>(dim, dim) {

    }
};

template <class T>
inline constexpr bool isC1Trans = std::is_convertible_v<T, TransC1>;

namespace Detail {

    template <class TransC1Impl>
    class GradientAsTransStub : public Stub<Trans, GradientAsTransStub<TransC1Impl>> {
    public:
        GradientAsTransStub(const TransC1Impl& impl)
            : Stub<Trans, GradientAsTransStub<TransC1Impl>>(impl.xdim(), impl.xdim() * impl.ydim())
              , trans_(impl) {

        }

        virtual Vec trans(const Vec& x, Vec to) const final {
            Mx toMx(to, trans_.xdim(), trans_.ydim());
            trans_.gradientTo(x, toMx);
            return to;
        }
    private:
        TransC1Impl trans_;
    };
}

template <class Impl>
class Stub<TransC1, Impl> : public virtual TransC1, public Stub<Trans, Impl> {
public:

    Mx gradientTo(const Vec& x, Mx to) const override {
        for (int64_t row = 0; row < ydim(); ++row) {
            static_cast<const Impl*>(this)->gradientRowTo(x, to.row(row), row);
        }
        return to;
    }

    std::unique_ptr<Trans> gradient() const override {
        return Detail::GradientAsTransStub<Impl>(*static_cast<const Impl*>(this));
    }

    Stub(int64_t xdim, int64_t ydim)
        : Stub<Trans, Impl>(xdim, ydim) {

    }
protected:

    std::unique_ptr<TransC1> cloneUniqueC1() const final {
        return std::unique_ptr<TransC1>(new Impl(*static_cast<const Impl*>(this)));
    }

    std::shared_ptr<TransC1> cloneSharedC1() const final {
        return std::make_shared<Impl>(*static_cast<const Impl*>(this));
    }
};

template <class Impl>
class MapC1Stub : public Stub<TransC1, Impl> {
public:
    MapC1Stub(int64_t dim)
        : Stub<TransC1, Impl>(dim, dim) {

    }
};

