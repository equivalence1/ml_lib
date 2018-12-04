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


class AnyFunc : public virtual AnyTrans {
public:
    virtual int64_t dim() const = 0;

    virtual int64_t ydim() const final {
        return 1;
    };

    double value(ConstVecRef x) const {
        Vec val(1);
        trans(x, val);
        return val.get(0);
    }

//    virtual Batch<VecRef> trans(Batch<ConstVec> x, Batch<VecRef> to) const = 0;
};

class AnyFuncC1 : public virtual AnyFunc, public virtual AnyTransC1 {
public:

    VecRef gradientRowTo(ConstVecRef x, VecRef to, int64_t index) const final {
        assert(index == 0);
        return gradientTo(x, to);
    }

};

namespace Detail {

    template <class T>
    class FuncWrapper : public AnyFunc {
    public:
        FuncWrapper(T&& impl)
            : instance_(std::move(impl)) {

        }

        int64_t dim() const final {
            return instance_.dim();
        }

        void valueTo(ConstVecRef x, DoubleRef to) const {
            instance_.value(x, to);
        }

//        Batch<Vec>& trans(const Batch<Vec>& x, Batch<Vec>& to) const final {
//            instance_.trans(x, to);
//            return to;
//        }

    private:
        T instance_;
    };

    template <class T>
    class FuncC1Wrapper : public AnyFuncC1 {
    public:
        FuncC1Wrapper(T&& impl)
            : instance_(std::move(impl)) {

        }

        FuncC1Wrapper(const T& impl)
            : instance_(impl) {

        }

        int64_t xdim() const final {
            return instance_.dim();
        }

        int64_t dim() const final {
            return instance_.dim();
        }

        VecRef trans(ConstVecRef x, VecRef to) const final {
            return instance_.trans(x, to);
        }

//        Batch<Vec>& trans(const Batch<Vec>& x, Batch<Vec>& to) const final {
//            instance_.trans(x, to);
//            return to;
//        }

        VecRef gradientTo(ConstVecRef x, VecRef to) const final {
            instance_.gradient().trans(x, to);
            return to;
        }

        ObjectPtr<AnyTrans> gradient() const {
            using GradTrans = decltype(instance_.gradient());
            using AnyTransImpl = TransWrapper<GradTrans>;
            return std::make_shared<AnyTransImpl>(instance_.gradient());
        }

    private:
        T instance_;
    };

}

namespace Detail {

    template <class Impl, class FuncInstance>
    class FuncBase {
    public:
        int64_t dim() const {
            return getInstance()->dim();
        }

        void valueTo(ConstVecRef x, DoubleRef to) const {
            return getInstance()->trans(x, to);
        }

//        Batch<VecRef> trans(Batch<ConstVecRef> x, Batch<VecRef> to) const {
//            return getInstance()->trans(x, to);
//        }

    protected:

        const FuncInstance* getInstance() const {
            return static_cast<Impl*>(this)->instance();
        }
    };

    template <class Impl, class FuncInstance>
    class Func1Base : public TransBase<Impl, FuncInstance> {
    public:
        using TransBase<Impl, FuncInstance>::getInstance;

        VecRef gradientTo(ConstVecRef x, VecRef to) const {
            getInstance()->gradientTo(x, to);
            return to;
        }
    };
}

class Func {
public:

    int64_t xdim() const {
        return dim();
    }

    int64_t ydim() const {
        return 1;
    }

    int64_t dim() const {
        return impl_->dim();
    }

    VecRef trans(ConstVecRef x, VecRef to) const {
        return impl_->trans(x, to);
    }

//    ConstVecRef trans(const Batch<Vec>& x, Batch<Vec>& to) const {
//        return impl_->trans(x, to);
//    }

    template <class T, class ... Args>
    static Func Create(Args&& ... args) {
        auto trans = std::make_shared<Detail::FuncWrapper<T>>(std::forward(args)...);
        return Func(std::static_pointer_cast<AnyFunc>(trans));
    }


    Trans asTrans() const {
        return Trans(std::static_pointer_cast<AnyTrans>(impl_));
    }
protected:

    Func(ObjectPtr<AnyFunc>&& impl)
    :impl_ (std::move(impl)) {

    }

    const AnyFunc* instance() const {
        return impl_.get();
    }

    ObjectPtr<AnyFunc> impl() const {
        return impl_;
    }
private:
    ObjectPtr<AnyFunc> impl_;

};

class FuncC1 : Func {
public:

    VecRef gradientTo(ConstVecRef x, VecRef to) const {
        this->c1instance()->gradientTo(x, to);
        return to;
    }

    Trans gradient() const {
        return Trans(this->c1instance()->gradient());
    }

    template <class T, class ... Args>
    static FuncC1 Create(Args&& ... args) {
        auto func = std::make_shared<Detail::FuncC1Wrapper<T>>(std::forward<Args>(args)...);
        return FuncC1(std::static_pointer_cast<AnyFuncC1>(func));
    }

    TransC1 asTransC1() const {
        auto asc1 = std::dynamic_pointer_cast<AnyTransC1>(impl());
        return TransC1(asc1);
    }
protected:

    FuncC1(ObjectPtr<AnyFuncC1> impl)
    : Func(std::static_pointer_cast<AnyFunc>(impl)) {

    }

    const AnyFuncC1* c1instance() const {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreinterpret-base-class"
        return reinterpret_cast<const AnyFuncC1*>(instance());
#pragma clang diagnostic pop
    }
};

template <class Impl>
class FuncStub {
public:
    FuncStub(int64_t dim)
    : dim_(dim) {

    }

    int64_t dim() const {
        return dim_;
    }

    operator Func() const {
        return asFunc();
    }

    operator Trans() const {
        return asFunc().ToTrans();
    }

    double value(ConstVecRef x) const {
        double result = 0;
        static_cast<const Impl*>(this)->valueTo(x, result);
        return result;
    }

    VecRef trans(ConstVecRef x, VecRef to) const {
        static_cast<const Impl*>(this)->trans(x, to);
        return to;
    }

private:
    Func asFunc() const {
        return Func::Create(*static_cast<const Impl*>(this));
    }
private:
    int64_t dim_;
};

template <class Impl>
class FuncC1Stub : public FuncStub<Impl> {
public:
    FuncC1Stub(int64_t dim)
    : FuncStub<Impl>(dim) {

    }

    operator FuncC1() const {
        return asFuncC1();
    }

    operator TransC1() const {
        return asFuncC1().ToTransC1();
    }


private:

    FuncC1 asFuncC1() const {
        return FuncC1::Create<Impl>(*static_cast<const Impl*>(this));
    }
};


