#include "boosting.h"
#include <models/ensemble.h>
#include <chrono>

ModelPtr Boosting::fit(const DataSet& dataSet, const Target& target)  {
    assert(&dataSet == &target.owner());
    Mx cursor(dataSet.samplesCount(),  1);
    std::vector<ModelPtr> models;


    auto start = std::chrono::system_clock::now();
    for (int32_t iter = 0; iter < config_.iterations_; ++iter) {
        auto weakTarget = weak_target_->create(dataSet, target, cursor);
        models.push_back(weak_learner_->fit(dataSet, *weakTarget)->scale(config_.step_));
        invoke(*models.back());
        models.back()->append(dataSet, cursor);
    }
    std::cout << "fit time " <<  std::chrono::duration<double>(std::chrono::system_clock::now() - start).count() << std::endl;

    return std::make_shared<Ensemble>(std::move(models));
}

Boosting::Boosting(
    const BoostingConfig& config,
    std::unique_ptr<EmpiricalTargetFactory>&& weak_target,
    std::unique_ptr<Optimizer>&& weak_learner)
    : config_(config)
    , weak_target_(std::move(weak_target))
    , weak_learner_(std::move(weak_learner)) {}