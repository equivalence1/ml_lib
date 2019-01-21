#include "load_data.h"

#include <torch/torch.h>
#include <core/vec_factory.h>

DataSet loadFeaturesTxt(const std::string& file) {
    std::ifstream in(file);

    std::vector<float> pool;
    std::vector<float> target;

    int64_t linesCount = 0;
    int64_t fCount = 0;
    std::string tempString;
    std::string line;

    while (std::getline(in, line) && line.size()) {
        std::istringstream parseTokens(line);
        float t = 0;
//qid
        parseTokens>>tempString;
//target
        parseTokens >>t;
//url
        parseTokens>>tempString;
//weight
        parseTokens>>tempString;

        parseTokens>>tempString;

        std::vector<float> lineFeatures(std::istream_iterator<float>{parseTokens},
                                        std::istream_iterator<float>());
        if (linesCount == 0) {
            fCount = lineFeatures.size();
        } else {
            assert(lineFeatures.size() == fCount);
        }

        pool.insert(pool.end(), lineFeatures.begin(), lineFeatures.end());
        target.push_back(t);
        ++linesCount;
    }

    auto data = VecFactory::create(ComputeDeviceType::Cpu, pool.size());
    ArrayRef<float> dst = data.arrayRef();
    std::copy(pool.begin(), pool.end(), dst.begin());


    auto targetVec = VecFactory::create(ComputeDeviceType::Cpu, target.size());
    std::copy(target.begin(), target.end(), targetVec.arrayRef().begin());

    Mx mx(data, linesCount, fCount);
    return DataSet(mx, targetVec);

}

