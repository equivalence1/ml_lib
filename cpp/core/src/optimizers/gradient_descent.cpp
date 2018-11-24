#include <core/optimizers/gradient_descent.h>

#include <core/vec_tools/transform.h>
#include <core/vec_tools/distance.h>

#include <cassert>

#include <stdio.h>

Vec GradientDescent::optimize(const Func& f, const Vec& x0) const {
    const double step = 0.1;
    double dist = eps_ + 1;

    auto grad_trans = f.gradient();

    Vec x1 = VecTools::copy(x0);
    Vec grad = VecFactory::create(VecType::Cpu, x0.dim());

    for (uint64_t iter = 0; iter < iter_lim_ && dist > eps_; iter++) {
        grad_trans->trans(x1, grad);
        for (int64_t i = 0; i < x0.dim(); i++) {
            x1.set(i, x1.get(i) - grad.get(i) * step);
        }
        dist = VecTools::norm(grad);
    }

    return x1;
}
