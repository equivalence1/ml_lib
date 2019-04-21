#include <memory>
#include <data/dataset.h>
#include <data/load_data.h>
#include <catboost_wrapper.h>
#include <models/polynom/polynom.h>

#define EPS 1e-5

inline TPool FromDataSet(const DataSet& ds) {
    TPool pool;
    pool.Features = ds.samples();
    pool.Labels = ds.labels();
    pool.FeaturesCount = ds.featuresCount();
    pool.SamplesCount = ds.samplesCount();
    return pool;
}
//
//
int main(int /*argc*/, char* /*argv*/[]) {

    auto ds = loadFeaturesTxt("test_data/featuresTxt/train");
    auto test = loadFeaturesTxt("test_data/featuresTxt/test");

    std::vector<float> learnData(ds.featuresCount() * ds.samplesCount());
    std::vector<float> testData(test.featuresCount() * test.samplesCount());
    for (size_t i = 0; i < ds.featuresCount(); ++i) {
        ds.visitColumn(i, [&](int lineIdx, float val) {
            learnData[i * ds.samplesCount() + lineIdx] = val;
        });
        test.visitColumn(i, [&](int lineIdx, float val) {
            testData[i * test.samplesCount() + lineIdx] = val;
        });
    }

    auto trainPool = FromDataSet(ds);
    trainPool.Features = learnData.data();
    auto testPool = FromDataSet(test);
    testPool.Features = testData.data();

    std::ifstream in("test_data/catboost_params.json");

    std::stringstream strStream;
    strStream << in.rdbuf(); //read the file
    std::string params = strStream.str();
    auto model = Train(trainPool, testPool, params);
    Polynom polynom(PolynomBuilder().AddEnsemble(model).Build());
    polynom.Lambda_ = 1000000;

    int outDim = 1;
    std::vector<float> out(outDim);
    double error = 0;
    for (int i = 0; i < testPool.SamplesCount; ++i) {
        polynom.Forward(ds.sample(i).arrayRef(), out);
        const double val = (out[0] - testPool.Labels[i]);
        error += val * val;
    }
    std::cout << "polynom error: " << error << std::endl;

}

