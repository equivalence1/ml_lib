#pragma once

#include "object.h"
#include <memory>
#include <functional>
#include <map>

template <class T>
class CacheHolder {
public:

    template <class From, class To, class Builder>
    const To& computeOrGet(std::shared_ptr<From> source, Builder&& builder) const {
        int64_t weakSource = source->uuid();// std::static_pointer_cast<Object>(source);
        if (!cache_.count(weakSource)) {
            cache_[weakSource] = std::unique_ptr<Object>(builder(*static_cast<const T*>(this), source).release());
        }

        auto result = cache_[weakSource].get();
        return *dynamic_cast<const To*>(result);
    }


private:
    mutable std::unordered_map<int64_t, std::unique_ptr<Object>> cache_;
};



