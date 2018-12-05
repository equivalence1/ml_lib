#pragma once

#include <memory>
#include <variant>
#include <optional>
#include <typeindex>
/*
* Base class for all RTTI classes
*/
class Object {
public:
    virtual ~Object() = default;

};

//easy to switch for intrusive, if we'll need it
template <class T>
using ObjectPtr = std::shared_ptr<T>;


template <class T>
using ObjectConstPtr = std::shared_ptr<const T>;

class AnyVec : public Object {
public:

    std::type_index id() const {
        if (!index_) {
            index_.emplace(typeid(*this));
        }
        return *index_;
    }

    template <class T>
    static inline std::type_index typeIndex() {
        return std::type_index(typeid(const T&));
    }
private:
    mutable std::optional<std::type_index> index_;

};


