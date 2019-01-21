#pragma once

#include "model.h"

class Ensemble : public Model {
public:

    Ensemble(std::vector<ModelPtr>&& models)
    : models_(std::move(models)) {

    }

    void append(const DataSet& ds, Vec* to) const override {
        for (const auto& model : models_) {
            model->append(ds, to);
        }
    }

    void apply(const DataSet& ds, Vec* to) const override {
        for (const auto& model : models_) {
            model->append(ds, to);
        }
    }

    int64_t size() const {
        return models_.size();
    }

private:

private:
    std::vector<std::unique_ptr<Model>> models_;
};
