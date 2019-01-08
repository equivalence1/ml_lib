#include <core/optimizers/momentum_grad.h>

#include <core/vec_tools/transform.h>
#include <core/vec_tools/distance.h>
#include <core/vec_tools/fill.h>

#include <cassert>
#include <stdio.h>

MomentumGrad::MomentumGrad(int64_t vecSize, double learningRate, double momentum)
    :
    learningRate(learningRate), momentum(momentum), gradBuf(vecSize) {
    VecTools::fill(0, gradBuf);
}

Vec MomentumGrad::optimize(Batch<FuncC1> f, Vec cursor) const {
    int64_t batchSize = f.size();
    Vec grad(cursor.dim());
    VecTools::fill(0, grad);
    Vec gradCur(cursor.dim());
    for (int64_t i = 0; i < batchSize; i++) {
        auto gradTrans = f[i].gradient();
        gradTrans.trans(cursor, gradCur);
        for (int64_t j = 0; j < cursor.dim(); j++) {
            grad.set(i, grad.get(i) + gradCur.get(i) / batchSize);
        }
    }
    for (int64_t i = 0; i < cursor.dim(); i++) {
        gradBuf.set(i, gradBuf.get(i) * momentum - learningRate * grad.get(i));
        cursor.set(i, cursor.get(i) + gradBuf.get(i));
    }
}

void MomentumGrad::reset() {
    VecTools::fill(0, gradBuf);
}
