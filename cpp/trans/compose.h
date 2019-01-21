#pragma once

#include <core/trans.h>

template <class Left, class Right>
inline constexpr bool isC1Composite = isC1Trans<Left> && isC1Trans<Right>;

template <class F, class G, bool C1 = isC1Composite<F, G>>
class ComposeTrans;

//F(G(X))
template <class F, class G>
class ComposeTrans<F, G, false> : public TransStub<ComposeTrans<F, G, false>> {
public:

    ComposeTrans(
        F f,
        G g)
        : TransStub<ComposeTrans>(g.xdim(), f.ydim())
          , f_(std::move(f))
          , g_(std::move(g)) {
        assert(f.xdim() == g.ydim());
    }

    Vec trans(const Vec& x, Vec to) const {
        Vec tmp(to.dim());
        g_.trans(x, tmp);
        f_.trans(tmp, to);
        return to;
    }

private:
    F f_;
    G g_;
};

//TOOD:
template <class F, class G>
class ComposeTrans<F, G, true> : public TransC1Stub<ComposeTrans<F, G, true>> {
public:

    ComposeTrans(
        F f,
        G g)
        : TransStub<ComposeTrans>(g.xdim(), f.ydim())
          , f_(std::move(f))
          , g_(std::move(g)) {
        assert(f.xdim() == g.ydim());
    }

    Vec trans(const Vec& x, Vec to) const {
        Vec tmp(to.dim());
        g_.trans(x, tmp);
        f_.trans(tmp, to);
        return to;
    }

//    CompositeGradient<F, G> gradient() const;

private:
    F f_;
    G g_;
};
