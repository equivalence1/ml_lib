#pragma once

#include "object.h"
#include "batch.h"
#include "vec.h"

#include <memory>
#include <optional>
#include <functional>

class Trans;
class TransC1;

//TODO(noxoomo): batch trans


class AnyTrans : public Object {
public:
    virtual int64_t xdim() const = 0;

    virtual int64_t ydim() const = 0;

    virtual Vec trans(const Vec& x, Vec to) const = 0;

//    virtual Batch<Vec> trans(Batch<ConstVec> x, Batch<Vec> to) const = 0;
};


class Trans  : public AnyTrans {
public:

    int64_t xdim() const final {
        return  impl_->xdim();
    }

    int64_t ydim() const final {
        return impl_->ydim();
    }

    Vec trans(const Vec& x, Vec to) const final {
        return impl_->trans(x, to);
    }

//    const Vec& trans(const Batch<Vec>& x, Batch<Vec>& to) const {
//        return Impl_->trans(x, to);
//    }



protected:
    friend class Func;
    friend class Func;
    friend class TransC1;

    template<class T, class ...Args>
    friend Trans CreateTrans(Args&&... args);

    Trans(ObjectPtr<AnyTrans>&& impl)
        : impl_(std::move(impl)) {

    }

    const AnyTrans* instance() const {
        return impl_.get();
    }
private:
    ObjectPtr<AnyTrans> impl_;
};


namespace Detail {

    template <class T>
    class TransWrapper : public AnyTrans {
    public:
        TransWrapper(T&& impl)
            : Instance_(std::move(impl)) {

        }

        int64_t xdim() const final {
            return Instance_.xdim();
        }

        int64_t ydim() const final {
            return Instance_.ydim();
        }

        Vec trans(const Vec& x, Vec to) const final {
            Instance_.trans(x, to);
            return to;
        }

//        Batch<Vec>& trans(const Batch<Vec>& x, Batch<Vec>& to) const final {
//            Instance_.trans(x, to);
//            return to;
//        }

    private:
        T Instance_;
    };
}

template <class T, class ... Args>
inline Trans CreateTrans(Args&&... args) {
    auto trans = std::make_shared<T>(std::forward<Args>(args)...);
    return Trans(std::static_pointer_cast<AnyTrans>(trans));
}



class AnyTransC1 : public virtual AnyTrans {
public:
    virtual Vec gradientTo(const Vec& x, Vec to) const = 0;

    virtual Vec gradientRowTo(const Vec& x, Vec to, int64_t index) const = 0;

    virtual Trans gradient() const = 0;
};



class TransC1 : public  AnyTransC1 {
public:

    operator Trans() const {
        return Trans(std::static_pointer_cast<AnyTrans>(impl_));
    }

    int64_t xdim() const final {
        return  impl_->xdim();
    }

    int64_t ydim() const final {
        return impl_->ydim();
    }

    Vec trans(const Vec& x, Vec to) const final {
        return impl_->trans(x, to);
    }
    //TOOD(noxoomo): make to Mx (Mx will be casted to to)
    //this one to func
    Vec gradientTo(const Vec& x, Vec to) const final {
        return impl_->gradientTo(x, to);
    }

    Vec gradientRowTo(const Vec& x, Vec to, int64_t index) const final {
        impl_->gradientRowTo(x, to, index);
        return to;
    }

    Trans gradient() const final {
       return impl_->gradient();
    }

protected:
    friend class FuncC1;

    TransC1(ObjectPtr<AnyTransC1> impl)
        : impl_(std::move(impl)) {

    }

private:
    ObjectPtr<AnyTransC1> impl_;
};


template <class T, class ... Args>
inline TransC1 CreateTransC1(Args&&... args) {
    auto trans = std::make_shared<T>(std::forward(args)...);
    return TransC1(std::static_pointer_cast<AnyTransC1>(trans));
}



template <class Impl>
class TransStub : public virtual AnyTrans {
public:
    operator Trans() const {
        return CreateTrans<Impl>(*static_cast<const Impl*>(this));
    }

    TransStub(int64_t xdim, int64_t ydim)
    : xdim_(xdim)
    , ydim_(ydim) {

    }

    int64_t xdim() const final {
        return xdim_;
    }

    int64_t ydim() const final {
        return ydim_;
    }

private:
    int64_t xdim_;
    int64_t ydim_;
};

template <class Impl>
class MapStub : public TransStub<Impl> {
public:
    MapStub(int64_t dim)
    : TransStub<Impl>(dim, dim) {

    }
};


template <class T>
inline constexpr bool isC1Trans = std::is_convertible_v<T, TransC1>;


template <class Impl>
class TransC1Stub : public TransStub<Impl>,  public AnyTransC1 {
public:
    operator TransC1() const {
        return CreateTransC1<Impl>(*static_cast<const Impl*>(this));
    }

    Vec gradientTo(const Vec& x, Vec to) const final {
        return static_cast<const Impl*>(this)->gradient().trans(x, to);
    }

    TransC1Stub(int64_t xdim, int64_t ydim)
    : TransStub<Impl>(xdim, ydim) {

    }
};


template <class Impl>
class MapC1Stub : public TransC1Stub<Impl> {
public:
    MapC1Stub(int64_t dim)
        : TransC1Stub<Impl>(dim, dim) {

    }
};

