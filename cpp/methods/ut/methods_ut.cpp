#include <memory>

#include <data/dataset.h>
#include <data/load_data.h>

#include <gtest/gtest.h>
#include <data/grid_builder.h>
#include <models/oblivious_tree.h>
#include <methods/boosting.h>
#include <methods/greedy_oblivious_tree.h>
#include <methods/boosting_weak_target_factory.h>
#include <targets/cross_entropy.h>
#include <metrics/accuracy.h>

#define EPS 1e-5
#define PATH_PREFIX "../../../../"

inline std::unique_ptr<GreedyObliviousTree> createWeakLearner(
    int32_t depth,
    GridPtr grid) {
    return std::make_unique<GreedyObliviousTree>(grid, depth);
}

inline std::unique_ptr<EmpiricalTargetFactory> createWeakTarget() {
    return std::make_unique<GradientBoostingWeakTargetFactory>();
}

inline std::unique_ptr<EmpiricalTargetFactory>  createBootstrapWeakTarget() {
    BootstrapOptions options;
    options.seed_ = 42;
    return std::make_unique<GradientBoostingBootstrappedWeakTargetFactory>(options);
}

TEST(FeaturesTxt, TestTrainMseFeaturesTxt) {

    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");
    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
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


TEST(FeaturesTxt, TestTrainWithBootstrapMseFeaturesTxt) {

    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");
    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLearner(6, grid));

    auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    metricsCalcer->addMetric(L2(test), "l2");
    boosting.addListener(metricsCalcer);
    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);

}




TEST(FeaturesTxt, TestTrainWithBootstrapLogLikelihoodFeaturesTxt) {

    auto ds = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/train");

    auto test = loadFeaturesTxt(PATH_PREFIX "test_data/featuresTxt/test");
    EXPECT_EQ(ds.samplesCount(), 12465);
    EXPECT_EQ(ds.featuresCount(), 50);

    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);

    BoostingConfig boostingConfig;
//    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLearner(6, grid));
    Boosting boosting(boostingConfig, createWeakTarget(), createWeakLearner(6, grid));

    auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    metricsCalcer->addMetric(CrossEntropy(test, 0.1), "CrossEntropy");
    metricsCalcer->addMetric(Accuracy(test.target(), 0.1, 0), "Accuracy");
    boosting.addListener(metricsCalcer);
    CrossEntropy target(ds, 0.1);
    auto ensemble = boosting.fit(ds, target);

}


//run it from root
TEST(FeaturesTxt, TestTrainMseMoscow) {
    auto start = std::chrono::system_clock::now();

    auto ds = loadFeaturesTxt("/Users/noxoomo/Projects/moscow_learn_200k.tsv");
    auto test = loadFeaturesTxt("/Users/noxoomo/Projects/moscow_test.tsv");
//    auto ds = loadFeaturesTxt("moscow_learn_200k.tsv");
//    auto test = loadFeaturesTxt("moscow_test.tsv");

    std::cout << " load data in memory " << std::endl;
    BinarizationConfig config;
    config.bordersCount_ = 32;
    auto grid = buildGrid(ds, config);
    std::cout << " build grid " << std::endl;

    BoostingConfig boostingConfig;
    Boosting boosting(boostingConfig, createBootstrapWeakTarget(), createWeakLearner(6, grid));

    auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
    metricsCalcer->addMetric(L2(test), "l2");
    boosting.addListener(metricsCalcer);
    L2 target(ds);
    auto ensemble = boosting.fit(ds, target);
    std::cout << "total time " << std::chrono::duration<double>(std::chrono::system_clock::now() - start).count()
              << std::endl;

}


