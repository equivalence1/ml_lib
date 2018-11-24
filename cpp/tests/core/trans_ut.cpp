#include <core/vec_factory.h>
#include <core/trans/offset_trans.h>
#include <core/trans/exp_trans.h>
#include <core/trans/linear_trans.h>
#include <core/trans/compose_trans.h>
#include <core/trans/const_trans.h>

#include <gtest/gtest.h>

#include <cmath>

// TODO EPS is so big because we store in float
#define EPS 1e-5

TEST(Trans, ConstTransTest) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, 123.0 * i / 3);
        b.set(i, i);
    }

    ConstTrans ct(b);
    ct.trans(a, a);

    for (auto i = 0; i < N; i++) {
        EXPECT_EQ(a(i), b(i));
    }
}

TEST(Trans, OffsetTransTest) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
        b.set(i, -i);
    }

    OffsetTrans off(b);
    off.trans(a, a);

    for (auto i = 0; i < N; i++) {
        EXPECT_EQ(a(i), 2 * i);
    }
}

TEST(Trans, ExpTransTest) {
    const int N = 10;
    const double exp = 2;

    Vec a = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
    }

    ExpTrans expT(exp, a.dim());
    expT.trans(a, a);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(a(i), std::pow(i, exp), EPS);
    }
}

TEST(Trans, LinearTransTest) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
        b.set(i, i);
    }

    LinearTrans l(b);
    l.trans(a, a);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(a(i), i * i, EPS);
    }
}

TEST(Trans, ComposeTransTest) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
        b.set(i, i);
    }

    auto l = std::make_shared<LinearTrans>(b);
    auto off = std::make_shared<OffsetTrans>(b);
    auto compose = std::make_shared<ComposeTrans>(off, l);

    compose->trans(a, a);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(a(i), i * i - i, EPS);
    }
}
