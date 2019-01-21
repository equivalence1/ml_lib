#include <core/vec.h>
#include <core/matrix.h>
#include <vec_tools/fill.h>

#include <gtest/gtest.h>

#include <memory>
#include <cmath>
#include <core/vec_factory.h>
#include <mx_tools/ops.h>

// TODO EPS is so big because we store in float
#define EPS 1e-5

// ops

TEST(MxOpsTest, Fill) {
    Vec vec = Vec(100);
    EXPECT_EQ(vec.dim(), 100);
    VecTools::fill(1, vec);
    Mx mx(vec, 10, 10);
    VecTools::fill(-1, mx);

    for (int32_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vec(i), -1);
        int32_t x = i % 10;
        int32_t y = i / 10;
        EXPECT_EQ(mx.get(x, y), -1);
    }

    VecTools::makeSequence(0, 1, mx);

    for (int32_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vec(i), i);
        int32_t x = i % 10;
        int32_t y = i / 10;
        EXPECT_EQ(mx.get(x, y), i);
    }

    Mx b(10, 10);
    VecTools::fill(10, b);

    mx += b;

    for (int32_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vec(i), i + 10);
        int32_t x = i % 10;
        int32_t y = i / 10;
        EXPECT_EQ(mx.get(x, y), i + 10);
    }

}

TEST(MxOpsTest, TestMultiply) {

    Vec vec(100);
    EXPECT_EQ(vec.dim(), 100);
    VecTools::makeSequence(0, 1.0 / 8, vec);
    Mx mx(10, 100);

    std::vector<double> refResult(10);
    {
        int64_t c = 0;
        for (int32_t i = 0; i < mx.ydim(); ++i) {
            for (int32_t j = 0; j < mx.xdim(); ++j) {
                const auto val = c * 1.34;
                ++c;
                mx.set(j, i, val);
                refResult[i] += val * vec.get(j);
            }
        }
    }
    Vec result(mx.ydim());
    MxTools::multiply(mx, vec, result);

    for (int32_t i = 0; i < mx.ydim(); ++i) {
        EXPECT_LE(std::abs(result(i) - refResult[i]), 1e-6);
    }

}



