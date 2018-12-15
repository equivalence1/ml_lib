#include <core/optimizers/gradient_descent.h>

#include <core/vec_tools/transform.h>
#include <core/vec_tools/distance.h>

#include <cassert>
#include <stdio.h>

Vec GradientDescent::optimize(FuncC1 f, Vec cursor) const {
    const double step = 0.1;
    double dist = eps_ + 1;
    auto gradTrans = f.gradient();
    Vec grad(cursor.dim());

    for (uint64_t iter = 0; iter < iter_lim_ && dist > eps_; iter++) {
        gradTrans.trans(cursor, grad);

        for (int64_t i = 0; i < cursor.dim(); i++) {
            cursor.set(i, cursor.get(i) - grad.get(i) * step);
        }
        dist = VecTools::norm(grad);
    }

    return cursor;
}
