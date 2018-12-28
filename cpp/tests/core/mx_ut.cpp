#include <core/vec.h>
#include <core/matrix.h>
#include <core/vec_tools/fill.h>

#include <gtest/gtest.h>

#include <memory>
#include <cmath>
#include <core/vec_factory.h>

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



