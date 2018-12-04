#include <core/vec.h>
#include <core/vec_factory.h>
#include <core/vec_tools/fill.h>
#include <core/vec_tools/distance.h>
#include <core/vec_tools/stats.h>
#include <core/vec_tools/transform.h>

#include <gtest/gtest.h>

#include <memory>
#include <cmath>

// TODO EPS is so big because we store in float
#define EPS 1e-5

// ops

TEST(OpsTest, Fill) {
    Vec vec = VecFactory::create(VecType::Cpu, 100);

    EXPECT_EQ(vec.dim(), 100);
    VecTools::fill(1, vec);
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vec(i), 1);
    }
}



#if defined(CUDA)
TEST(OpsTest, FillGpu) {
    const auto n = 10000;
    Vec vec = VecFactory::create(VecType::Gpu, n);

    EXPECT_EQ(vec.dim(), n);
    VecTools::fill(1, vec);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(vec(i), 1);
    }
}
#endif

TEST(OpsTest, DotProduct) {
    Vec a = VecFactory::create(VecType::Cpu, 10);
    Vec b = VecFactory::create(VecType::Cpu, 10);


    EXPECT_EQ(a.dim(), 10);
    EXPECT_EQ(b.dim(), 10);
    VecTools::fill(1, a);
    VecTools::makeSequence(0.0, 1.0, b);
    double res = VecTools::dotProduct(a, b);
    EXPECT_NEAR(res, 45, EPS);
}

TEST(OpsTest, Subtract) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);

    for (auto i = 0; i < N; i++) {
        a.set(i, i);
        b.set(i, 2 * i);
    }

    Vec& res = VecTools::subtract(a, b);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(res(i), -i, EPS);
    }
}

TEST(OpsTest, Exp) {
    const int N = 10;
    const double exp = 2.33;

    Vec a = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
    }

    Vec b = VecFactory::create(VecType::Cpu, N);
    VecTools::pow(exp, a, b);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(b(i), std::pow(i, exp), EPS);
    }
}

TEST(OpsTest, Mul) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
        b.set(i, i);
    }

    VecTools::mul(a, b);

    for (auto i = 0; i < N; i++) {
        EXPECT_NEAR(b(i), i * i, EPS);
    }
}

// stats

TEST(StatsTest, Sum) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, 3 * i);
    }

    double res = VecTools::sum(a);
    EXPECT_NEAR(res, 3 * N * (N - 1) / 2, EPS);
}

TEST(StatsTest, Sum2) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, 3 * i);
    }

    double res = VecTools::sum2(a);
    EXPECT_NEAR(res, 9 * N * (N - 1) * (2 * (N - 1) + 1) / 6, EPS);
}

// distance

TEST(DistanceTest, Norm) {
    Vec a = VecFactory::create(VecType::Cpu, 2);
    a.set(0, 3);
    a.set(1, 4);

    double res = VecTools::norm(a);
    EXPECT_NEAR(res, 5, EPS);
}

TEST(DistanceTest, DistanceL1) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);

    for (auto i = 0; i < N; i++) {
        if (i % 2) {
            a.set(i, i);
            b.set(i, 2 * i);
        } else {
            a.set(i, 2 * i);
            b.set(i, i);
        }
    }

    double res = VecTools::distanceL1(a, b);
    double expected_res = N * (N - 1) / 2;

    EXPECT_NEAR(res, expected_res, EPS);
}

TEST(DistanceTest, DistanceL2) {
    Vec a = VecFactory::create(VecType::Cpu, 4);
    a.set(0, 1);
    a.set(1, 0);
    a.set(2, 2);
    a.set(3, -1);

    Vec b = VecFactory::create(VecType::Cpu, 4);
    b.set(0, 4);
    b.set(1, 0);
    b.set(2, 6);
    b.set(3, -1);

    double res = VecTools::distanceL2(a, b);
    EXPECT_NEAR(res, 5, EPS);
}

// transform

TEST(TransformTest, Copy) {
    const int N = 10;

    Vec a = VecFactory::create(VecType::Cpu, N);
    Vec b = VecFactory::create(VecType::Cpu, N);
    for (auto i = 0; i < N; i++) {
        a.set(i, i);
    }

    VecTools::copyTo(a, b);
    for (auto i = 0; i < N; i++) {
        EXPECT_EQ(a(i), b(i));
    }
}
