#include <gtest/gtest.h>
#include <core/vec.h>
#include <core/vec_factory.h>
#include <core/vec_tools/fill.h>

TEST(OpsTest, Fill) {
    Vec vec = VecFactory::create(VecType::Cpu, 100);

    EXPECT_EQ(vec.dim(), 100);
    VecTools::fill(1, vec);
    for (int64_t i = 0; i < 100; ++i) {
        EXPECT_EQ(vec(i), 1);
    }
}



TEST(OpsTest, FillGpu) {
    const auto n = 10000;
    Vec vec = VecFactory::create(VecType::Gpu, n);

    EXPECT_EQ(vec.dim(), n);
    VecTools::fill(1, vec);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_EQ(vec(i), 1);
    }
}

TEST(OpsTest, DotProduct) {
    Vec a = VecFactory::create(VecType::Cpu, 10);
    Vec b = VecFactory::create(VecType::Cpu, 10);


    EXPECT_EQ(a.dim(), 10);
    EXPECT_EQ(b.dim(), 10);
    VecTools::fill(1, a);
    VecTools::makeSequence(0.0, 1.0, b);
    double res = VecTools::dotProduct(a, b);
    EXPECT_EQ(res, 45);
}


