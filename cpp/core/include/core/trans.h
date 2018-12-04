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

    virtual VecRef trans(ConstVecRef x, VecRef to) const = 0;

//    virtual Batch<VecRef> trans(Batch<ConstVec> x, Batch<VecRef> to) const = 0;
};



class AnyTransC1 : public virtual AnyTrans {
public:
    virtual VecRef gradientTo(ConstVecRef x, VecRef to) const = 0;

    virtual VecRef gradientRowTo(ConstVecRef x, VecRef to, int64_t index) const = 0;

    virtual ObjectPtr<AnyTrans> gradient() const = 0;

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

        VecRef trans(ConstVecRef x, VecRef to) const final {
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

    template <class T>
    class TransC1Wrapper : public AnyTransC1 {
    public:
        TransC1Wrapper(T&& impl)
            : Instance_(std::move(impl)) {

        }

        int64_t xdim() const final {
            return Instance_.xdim();
        }

        int64_t ydim() const final {
            return Instance_.ydim();
        }

        VecRef trans(ConstVecRef x, VecRef to) const final {
            Instance_.trans(x, to);
            return to;
        }

//        Batch<Vec>& trans(const Batch<Vec>& x, Batch<Vec>& to) const final {
//            Instance_.trans(x, to);
//            return to;
//        }

        VecRef gradientTo(ConstVecRef x, VecRef to) const final {
            Instance_.gradientTo(x, to);
            return to;
        }

        VecRef gradientRowTo(ConstVecRef x, VecRef to, int64_t index) const {
            return Instance_.gradientRowTo(x, to, index);
        }

        ObjectPtr<AnyTrans> gradient() const {
            using GradTrans = decltype(Instance_.gradient());
            using AnyTransImpl = TransWrapper<GradTrans>;
            return std::make_shared<AnyTransImpl>(Instance_.gradient());
        }

    private:
        T Instance_;
    };

}

namespace Detail {

    template <class Impl, class TransInstance>
    class TransBase {
    public:
        int64_t xdim() const {
            return getInstance()->xdim();
        }

        int64_t ydim() const {
            return getInstance()->ydim();
        }

        ConstVec trans(ConstVecRef x, VecRef to) const {
            return getInstance()->trans(x, to);
        }

//        Batch<VecRef> trans(Batch<ConstVecRef> x, Batch<VecRef> to) const {
//            return getInstance()->trans(x, to);
//        }

    protected:

        const TransInstance* getInstance() const {
            return static_cast<Impl*>(this)->instance();
        }
    };

    template <class Impl, class TransInstance>
    class TransC1Base : public TransBase<Impl, TransInstance> {
    public:
        using TransBase<Impl, TransInstance>::getInstance;

        VecRef gradientTo(ConstVecRef x, VecRef to) const {
            getInstance()->gradientTo(x, to);
            return to;
        }

        VecRef gradientRowTo(ConstVecRef x, VecRef to, int64_t index) const {
            getInstance()->gradientRowTo(x, to, index);
            return to;
        }
    };
}


class Trans  {
public:

    int64_t xdim() const {
        return  Impl_->xdim();
    }

    int64_t ydim() const {
        return Impl_->ydim();
    }

    VecRef trans(ConstVecRef x, VecRef to) const {
        return Impl_->trans(x, to);
    }

//    ConstVecRef trans(const Batch<Vec>& x, Batch<Vec>& to) const {
//        return Impl_->trans(x, to);
//    }

    template <class T, class ... Args>
    static Trans Create(Args&&... args) {
        auto trans = std::make_shared<Detail::TransWrapper<T>>(std::forward(args)...);
        return Trans(std::static_pointer_cast<AnyTrans>(trans));
    }

protected:
    friend class Func;
    friend class FuncC1;

    Trans(ObjectPtr<AnyTrans>&& impl)
        : Impl_(std::move(impl)) {

    }

    const AnyTrans* instance() const {
        return Impl_.get();
    }
private:
    ObjectPtr<AnyTrans> Impl_;

};


class TransC1 : Trans {
public:
    //TOOD(noxoomo): make to Mx (Mx will be casted to to)
    //this one to func
    VecRef gradientTo(ConstVecRef x, VecRef to) const {
        this->c1instance()->gradientTo(x, to);
        return to;
    }

    VecRef gradientRowTo(ConstVecRef x, VecRef to, int64_t index) const {
        this->c1instance()->gradientRowTo(x, to, index);
        return to;
    }

    template <class T, class ... Args>
    static TransC1 Create(Args&&... args) {
        auto trans = std::make_shared<Detail::TransC1Wrapper<T>>(std::forward(args)...);
        return TransC1(std::static_pointer_cast<AnyTransC1>(trans));
    }
protected:
    friend class FuncC1;

    TransC1(const ObjectPtr<AnyTransC1>& impl)
        : Trans(std::static_pointer_cast<AnyTrans>(impl)) {

    }

    const AnyTransC1* c1instance() const {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreinterpret-base-class"
        return reinterpret_cast<const AnyTransC1*>(instance());
#pragma clang diagnostic pop
    }
};

template <class Impl>
class TransStub {
public:
    operator Trans() const {
        return Trans::Create(*static_cast<const Impl*>(this));
    }

    TransStub(int64_t xdim, int64_t ydim)
    : xdim_(xdim)
    , ydim_(ydim) {

    }

    int64_t xdim() const {
        return xdim_;
    }

    int64_t ydim() const {
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
class TransC1Stub : public TransStub<Impl> {
public:
    operator TransC1() const {
        return TransC1::Create(*static_cast<const Impl*>(this));
    }

    VecRef gradientTo(ConstVecRef x, VecRef to) const {
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

