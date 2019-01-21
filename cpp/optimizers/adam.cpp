#include <core/optimizers/adam.h>

#include <core/vec_tools/transform.h>
#include <core/vec_tools/distance.h>
#include <core/vec_tools/fill.h>

#include <cassert>
#include <stdio.h>
#include <cmath>

Adam::Adam(int64_t vecSize, double learningRate, double betta1, double betta2, double eps)
    :
    betta1(betta1), betta2(betta2), learningRate(learningRate), eps(eps), betta1Pow(1), betta2Pow(1), vD(0), sD(0) {
    VecTools::fill(0, vD);
    VecTools::fill(0, sD);
}

Vec Adam::optimize(Batch<FuncC1> f, Vec cursor) const {
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

    betta1Pow *= betta1;
    betta2Pow *= betta2;

    for (int64_t i = 0; i < cursor.dim(); i++) {
        vD.set(i, betta1 * vD.get(i) + (1 - betta1) * grad.get(i));
        sD.set(i, betta2 * sD.get(i) + (1 - betta2) * (grad.get(i) * grad.get(i)));
        cursor.set(i,
                   cursor.get(i)
                       - learningRate * vD.get(i) / ((1 - betta1Pow) * (std::sqrt(sD.get(i) / (1 - betta2Pow)) + eps)));

    }
}

void Adam::reset() {
    VecTools::fill(0, vD);
    VecTools::fill(0, sD);
    betta1Pow = betta1;
    betta2Pow = betta2;
}
