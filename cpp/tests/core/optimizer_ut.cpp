#include <core/vec_factory.h>
#include <core/optimizers/gradient_descent.h>
#include <core/vec_tools/fill.h>
#include <core/vec_tools/distance.h>
#include <core/funcs/lq.h>

#include <gtest/gtest.h>

TEST(Optimizer, GradientDescentTest) {
    const int N = 10;
    const double q = 2;
    const double EPS = 0.01;

    GradientDescent gd(EPS);

    auto cursor = Vec(N);
    VecTools::fill(1, cursor);

    auto b = Vec(N);
    VecTools::fill(2, b);

    Lq distFunc(q, b);

    gd.optimize(distFunc, cursor);

    EXPECT_LE(VecTools::distanceLq(q, cursor, b), EPS);
}
