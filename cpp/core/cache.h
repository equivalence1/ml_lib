#pragma once

#include "object.h"
#include <memory>
#include <functional>
#include <map>


//TODO(noxoomo): this is very optimistic, fix it
struct ObjectWeakPtr {

    ObjectWeakPtr(std::shared_ptr<Object> ptr)
        : ptr_(ptr)
          , hash_(std::hash<uint64_t>()(reinterpret_cast<uint64_t>(ptr.get())))
          , raw_ptr_(ptr.get()) {

    }

    std::weak_ptr<Object> ptr_;
    uint64_t hash_;
    Object* raw_ptr_;

    bool expired() const {
        return ptr_.expired();
    }

    bool operator<(const ObjectWeakPtr& rhs) const {
        return raw_ptr_ < rhs.raw_ptr_;
    }

    bool operator ==(const ObjectWeakPtr& rhs) const  {
        return raw_ptr_ == rhs.raw_ptr_;
    }
};


namespace std {
    template <>
    struct hash<ObjectWeakPtr> {

        size_t operator()(const ObjectWeakPtr& ptr) const {
            return ptr.hash_;
        }

    };
}
template <class T>
class CacheHolder {
public:


    template <class From, class To, class Builder>
    const To& computeOrGet(std::shared_ptr<From> source, Builder&& builder) const {
        uint64_t weakSource = (uint64_t)(source.get());// std::static_pointer_cast<Object>(source);
        if (!cache_.count(weakSource)) {
            cache_[weakSource] = std::unique_ptr<Object>(builder(*static_cast<const T*>(this), source).release());
        }

        auto result = cache_[weakSource].get();
        return *dynamic_cast<const To*>(result);
    }

//    void clearExpired() const {
//        std::unordered_map<ObjectWeakPtr, std::shared_ptr<Object>> notExpired;
//        for (auto [keyPtr, valuePtr] : cache_) {
//            if (!keyPtr.expired()) {
//                notExpired[keyPtr] = std::move(valuePtr);
//            }
//        }
//        cache_.swap(notExpired);
//    }

private:
    mutable std::unordered_map<uint64_t, std::unique_ptr<Object>> cache_;
};



