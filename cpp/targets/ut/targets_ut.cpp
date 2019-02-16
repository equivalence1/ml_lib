#include <data/dataset.h>
#include <data/load_data.h>

#include <core/vec.h>

#include <gtest/gtest.h>
#include <data/grid_builder.h>
#include <data/binarized_dataset.h>
#include <models/oblivious_tree.h>
#include <targets/l2.h>

#define EPS 1e-5

//run it from root
TEST(TargetsTest, TestL2) {
    auto ds = loadFeaturesTxt("test_data/featuresTxt/train");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);
    L2 target(ds);

    Vec cursor(target.dim());
    for (int64_t i = 0; i < cursor.dim(); ++i) {
        cursor.set(i, i * 1.0 / target.dim());
    }
    Vec der(target.dim());
    target.gradientTo(cursor, der);

    for (int64_t i = 0; i < target.dim(); ++i) {
        EXPECT_LE(std::abs(der.get(i) - (ds.target().get(i) - i * 1.0 / target.dim())), 1e-5);
    }


}



