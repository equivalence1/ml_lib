#include <core/funcs/p_dist_func.h>

#include <core/vec.h>
#include <core/vec_factory.h>
#include <core/vec_tools/distance.h>
#include <core/trans/linear_trans.h>
#include <core/trans/exp_trans.h>
#include <core/trans/compose_trans.h>
#include <core/trans/offset_trans.h>
#include <core/vec_tools/fill.h>

int64_t PDistFunc::xdim() const {
    return b_.dim();
}

double PDistFunc::value(const Vec& x) const {
    return VecTools::distanceP(p_, x, b_);
}

ObjectPtr<Trans> PDistFunc::gradient() const {
    Vec p = VecFactory::create(VecType::Cpu, b_.dim());
    VecTools::fill(p_, p);
    auto linear = std::make_shared<LinearTrans>(p);
    auto exp = std::make_shared<ExpTrans>(p_ - 1, b_.dim());
    auto offset = std::make_shared<OffsetTrans>(b_);
    auto exp_off = std::make_shared<ComposeTrans>(exp, offset);
    auto res = std::make_shared<ComposeTrans>(linear, exp_off);
    return res;
}
