#pragma once

#include <core/object.h>
#include <core/func.h>
#include <vector>
#include <memory>

template <class T>
class Listener : public Object {
public:
    virtual void operator()(const T& event) = 0;
};

template <class T>
class ListenersHolder : public Object {
public:
    using Inner = Listener<T>;

    void addListener(SharedPtr<Inner> listener) {
        listeners_.push_back(listener);
    }

protected:

    void invoke(const T& event) const {
        for (int64_t i = 0; i < listeners_.size(); ++i) {
            if (!listeners_[i].expired()) {
                SharedPtr<Inner> ptr = listeners_[i].lock();
                (*ptr)(event);
            }
        }
    }

private:

    std::vector<std::weak_ptr<Inner>> listeners_;
};


class BoostingMetricsCalcer : public Listener<Model> {
public:

    BoostingMetricsCalcer(const DataSet& ds)
    : ds_(ds)
    , cursor_(ds.target().dim(), 1) {

    }

    void operator()(const Model& model) override {
        model.append(ds_, cursor_);
        if (iter_ % 50 == 1) {
            std::cout << "iter " << iter_<<": ";
            for (int32_t i = 0; i < metrics_.size(); ++i) {
                std::cout << metricName[i] << "=" << metrics_[i]->value(cursor_);
                if (i + 1 != metrics_.size()) {
                    std::cout << "\t";
                }
            }
            std::cout << std::endl;
        }
        ++iter_;
    }

    void addMetric(const Func& func, const std::string& name) {
        metrics_.push_back(func);
        metricName.push_back(name);
    }
private:
    std::vector<SharedPtr<Func>> metrics_;
    std::vector<std::string> metricName;
    const DataSet& ds_;
    Mx cursor_;
    int32_t iter_ = 0;
};


class IterPrinter : public Listener<Model> {
public:

    IterPrinter() {

    }

    void operator()(const Model& model) override {
        if (iter_ % 10 == 0) {
            std::cout << "iter " << iter_<<std::endl;
        }
        ++iter_;
    }

private:
    int32_t iter_ = 0;
};
