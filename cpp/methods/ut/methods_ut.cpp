#include <memory>

#include <data/dataset.h>
#include <data/load_data.h>

#include <core/vec.h>

#include <gtest/gtest.h>
#include <data/grid_builder.h>
#include <data/binarized_dataset.h>
#include <models/oblivious_tree.h>
#include <methods/boosting.h>
#include <methods/greedy_oblivious_tree.h>
#include <methods/boosting_weak_target_factory.h>

#define EPS 1e-5


inline std::unique_ptr<GreedyObliviousTree> createWeakLearner(int32_t depth,
                                                              GridPtr grid) {
    return std::make_unique<GreedyObliviousTree>(grid, depth);
}


inline std::unique_ptr<EmpiricalTargetFactory> createWeakTarget() {
    return std::make_unique<GradientBoostingWeakTargetFactory>();

}

//run it from root
TEST(FeaturesTxt, TestTrainMseFeaturesTxt) {

        auto ds = loadFeaturesTxt("test_data/featuresTxt/train");
        auto test = loadFeaturesTxt("test_data/featuresTxt/test");
        EXPECT_EQ(ds.samplesCount(), 12465);
        EXPECT_EQ(ds.featuresCount(), 50);

        BinarizationConfig config;
        config.bordersCount_ = 32;
        auto grid = buildGrid(ds, config);

        BoostingConfig boostingConfig;
        Boosting boosting(boostingConfig, createWeakTarget(), createWeakLearner(6, grid));

        auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
        metricsCalcer->addMetric(L2(test), "l2");
        boosting.addListener(metricsCalcer);
        L2 target(ds);
        auto ensemble = boosting.fit(ds, target);




}



