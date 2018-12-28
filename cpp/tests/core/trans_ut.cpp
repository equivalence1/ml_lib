#include <core/vec_factory.h>
#include <core/trans/add_vec.h>
#include <core/trans/pow.h>
#include <core/trans/pointwise_multiply.h>
#include <core/trans/compose.h>
#include <core/trans/fill.h>

#include <gtest/gtest.h>

#include <cmath>

// TODO EPS is so big because we store in float
#define EPS 1e-5

TEST(Trans, FillTransTest) {
    const int N = 10;

    Vec a = Vec(N);
    Vec b = Vec(N);
    Vec c = Vec(N);
    for (auto i = 0; i < N; i++) {
        a.set(i, 123.0 * i / 3);
        b.set(i, i);
        c.set(i, 100500);
    }

    FillVec ct(b, a.dim());
    ct.trans(a, c);

    for (auto i = 0; i < N; i++) {
        EXPECT_EQ(c(i), b(i));
    }
}

TEST(Trans, AddVecTest) {
    const int N = 10;

    Vec a = Vec(N);
    Vec b = Vec(N);
    Vec c = Vec(N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
        b.set(i, -i);
        c.set(i, 100500);
    }

    AddVecTrans off(b);
    off.trans(a, c);

    for (auto i = 0; i < N; i++) {
        EXPECT_EQ(c(i), 2 * i);
    }
}

TEST(Trans, PowTest) {
    const int N = 10;
    const double exp = 2;

    Vec a = Vec(N);
    Vec b = Vec(N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i * 1.0 / N);
        b.set(i, 100500);
    }

    Pow pow(exp, a.dim(), 3);
    pow.trans(a, b);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(b(i), 3 * std::pow(i * 1.0 / N, exp), EPS);
    }
}

TEST(Trans, PointwiseMulTest) {
    const int N = 10;

    Vec a = Vec(N);
    Vec b = Vec(N);
    Vec c = Vec(N);
    for (auto i = 0; i < N; i++) {
        a.set(i, 1.0 / (i + 1));
        b.set(i, i + 1);
        c.set(i, 100500);
    }

    PointwiseMultiply l(b);
    l.trans(a, c);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(c(i), 1.0, EPS);
    }
}

//
//TEST(Trans, ComposeTransTest) {
//    const int N = 10;
//
//    Vec a = VecFactory::create(VecType::Cpu, N);
//    Vec b = VecFactory::create(VecType::Cpu, N);
//    for (auto i = 0; i < N; i++) {
//        a.set(i, i);
//        b.set(i, i);
//    }
//
//    auto l = std::make_shared<LinearTrans>(b);
//    auto off = std::make_shared<OffsetTrans>(b);
//    auto compose = std::make_shared<ComposeTrans>(off, l);
//
//    compose->trans(a, a);
//
//    for (auto i = 0; i < N; i++) {
//        EXPECT_NEAR(a(i), i * i - i, EPS);
//    }
//}
