#include <core/vec_factory.h>
#include <core/optimizers/gradient_descent.h>
#include <core/vec_tools/fill.h>
#include <core/vec_tools/distance.h>
#include <core/funcs/p_dist_func.h>

#include <gtest/gtest.h>

TEST(Optimizer, GradientDescentTest) {
    const int N = 10;
    const double P = 2;
    const double EPS = 0.01;

    GradientDescent gd(EPS);

    auto a = VecFactory::create(VecType::Cpu, N);
    VecTools::fill(1, a);

    auto b = VecFactory::create(VecType::Cpu, N);
    VecTools::fill(2, b);

    auto distFunc = PDistFunc(P, b);

    Vec res = gd.optimize(distFunc, a);

    EXPECT_LE(VecTools::distanceP(P, res, b), EPS);
}
