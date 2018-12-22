#include <core/optimizers/stochastic_grad.h>

#include <core/vec_tools/transform.h>
#include <core/vec_tools/distance.h>
#include <core/vec_tools/fill.h>

#include <cassert>
#include <stdio.h>

Vec StochasticGrad::optimize(Batch<FuncC1> f, Vec cursor) const {
    int64_t batchSize = f.size();
    Vec grad(cursor.dim());
    VecTools::fill(0, grad);
    Vec gradCur(cursor.dim());
    for (uint64_t i = 0; i < batchSize; i++) {
        auto gradTrans = f[i].gradient();
        gradTrans.trans(cursor, gradCur);
        for (int64_t j = 0; j < cursor.dim(); j++) {
            grad.set(i, grad.get(i) + gradCur.get(i) / batchSize);
        }
    }
    for (uint64_t i = 0; i < cursor.dim(); i++) {
        cursor.set(i, cursor.get(i) - learningRate * grad.get(i));
    }
}