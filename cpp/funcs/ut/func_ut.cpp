#include <core/vec.h>
#include <funcs/linear.h>
#include <funcs/lq.h>

#include <gtest/gtest.h>

#define EPS 1e-5

TEST(FuncTests, Linear) {
    Vec param = Vec(2);

    double bias = 1;
    param.set(0, -2);
    param.set(1, 3);

    Vec x = Vec(2);
    x.set(0, 10);
    x.set(1, 20);

    Linear linear(param, bias);
    EXPECT_EQ(linear.dim(), 2);
    double res = linear.value(x);
    EXPECT_EQ(res, 41);

}

TEST(FuncTests, Lq) {
    const double p = 2;

    Vec x = Vec(3);
    x.set(0, -1);
    x.set(1, 2);
    x.set(2, 3);

    Vec b = Vec(3);
    b.set(0, -1);
    b.set(1, 5);
    b.set(2, 7);

    Lq d(p, b);
    auto res = d.value(x);
    EXPECT_EQ(res, 5);

    Vec c = Vec(3);
    auto grad = d.gradient();
    grad->trans(x, c);

    for (auto i = 0; i < 3; i++) {
        EXPECT_NEAR(c(i), p * std::pow(x(i) - b(i), p - 1), EPS);
    }
}
