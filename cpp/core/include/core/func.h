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
//
//namespace Detail {
//
//    template <class T>
//    class FuncWrapper : public AnyFunc {
//    public:
//        FuncWrapper(T&& impl)
//            : instance_(std::move(impl)) {
//
//        }
//
//        int64_t dim() const final {
//            return instance_.dim();
//        }
//
//        void valueTo(ConstVecRef x, DoubleRef to) const {
//            instance_.value(x, to);
//        }
//
////        Batch<Vec>& trans(const Batch<Vec>& x, Batch<Vec>& to) const final {
////            instance_.trans(x, to);
////            return to;
////        }
//
//    private:
//        T instance_;
//    };
//
//    template <class T>
//    class FuncC1Wrapper : public AnyFuncC1 {
//    public:
//        FuncC1Wrapper(T&& impl)
//            : instance_(std::move(impl)) {
//
//        }
//
//        FuncC1Wrapper(const T& impl)
//            : instance_(impl) {
//
//        }
//
//        int64_t xdim() const final {
//            return instance_.dim();
//        }
//
//        int64_t dim() const final {
//            return instance_.dim();
//        }
//
//        VecRef trans(ConstVecRef x, VecRef to) const final {
//            return instance_.trans(x, to);
//        }
//
////        Batch<Vec>& trans(const Batch<Vec>& x, Batch<Vec>& to) const final {
////            instance_.trans(x, to);
////            return to;
////        }
//
//        VecRef gradientTo(ConstVecRef x, VecRef to) const final {
//            instance_.gradient().trans(x, to);
//            return to;
//        }
//
//        Trans gradient() const {
//            using GradTrans = decltype(instance_.gradient());
//            using AnyTransImpl = TransWrapper<GradTrans>;
//            return std::make_shared<AnyTransImpl>(instance_.gradient());
//        }
//
//    private:
//        T instance_;
//    };
//}


class Func : public AnyFunc {
public:

    int64_t xdim() const final {
        return dim();
    }

    int64_t dim() const final {
        return impl_->dim();
    }

    VecRef trans(ConstVecRef x, VecRef to) const final {
        return impl_->trans(x, to);
    }

//    ConstVecRef trans(const Batch<Vec>& x, Batch<Vec>& to) const {
//        return impl_->trans(x, to);
//    }

    operator Trans() const {
        return asTrans();
    }
protected:
    template <class T, class ... Args>
    friend Func CreateFunc(Args&& ... args);

    Trans asTrans() const {
        return Trans(std::static_pointer_cast<AnyTrans>(impl_));
    }

    Func(ObjectPtr<AnyFunc>&& impl)
        : impl_(std::move(impl)) {

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


template <class T, class ... Args>
inline Func CreateFunc(Args&& ... args) {
    auto trans = std::make_shared<T>(std::forward(args)...);
    return Func(std::static_pointer_cast<AnyFunc>(trans));
}



class FuncC1 : public AnyFuncC1 {
public:

    int64_t xdim() const final {
        return dim();
    }

    int64_t dim() const final {
        return impl_->dim();
    }

    VecRef trans(ConstVecRef x, VecRef to) const final {
        return impl_->trans(x, to);
    }

    VecRef gradientTo(ConstVecRef x, VecRef to) const {
        return impl_->gradientTo(x, to);
    }

    Trans gradient() const {
        return impl_->gradient();
    }



    operator TransC1() const {
        return asTransC1();
    }
protected:
    template <class T, class... Args>
    friend FuncC1 CreateFuncC1(Args&& ... args);

    TransC1 asTransC1() const {
        auto asc1 = std::dynamic_pointer_cast<AnyTransC1>(impl_);
        return TransC1(asc1);
    }

    FuncC1(ObjectPtr<AnyFuncC1> impl)
        : impl_(std::move(impl)) {

    }

private:
    ObjectPtr<AnyFuncC1> impl_;
};




template <class T, class ... Args>
inline FuncC1 CreateFuncC1(Args&& ... args) {
    auto func = std::make_shared<T>(std::forward<Args>(args)...);
    return FuncC1(std::static_pointer_cast<AnyFuncC1>(func));
}

template <class Impl>
class FuncStub : public virtual AnyFunc {
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

    VecRef trans(ConstVecRef x, VecRef to) const final {
        static_cast<const Impl*>(this)->trans(x, to);
        return to;
    }

private:
    Func asFunc() const {
        return CreateFunc<Impl>(*static_cast<const Impl*>(this));
    }
private:
    int64_t dim_;
};

template <class Impl>
class FuncC1Stub : public FuncStub<Impl>, virtual public AnyFuncC1 {
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

    VecRef gradientTo(ConstVecRef x, VecRef to) const {
        return static_cast<const Impl*>(this)->gradient().trans(x, to);
    }

private:

    FuncC1 asFuncC1() const {
        return CreateFuncC1<Impl>(*static_cast<const Impl*>(this));
    }
};


