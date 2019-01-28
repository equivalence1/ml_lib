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

//    virtual size_t hash() const = 0;
//    virtual bool equals(const Object& other) const = 0;
};



//easy to switch for intrusive, if we'll need it
template <class T>
using SharedPtr = std::shared_ptr<T>;


//easy to switch for intrusive, if we'll need it
template <class T>
using UniquePtr = std::unique_ptr<T>;

template <class T>
using SharedConstPtr = std::shared_ptr<const T>;

//
//
//namespace std {
//
//    template <>
//    struct std::hash<SharedPtr<Object>> {
//        size_t operator()(const SharedPtr<Object>& obj) {
//            return obj->hash();
//        }
//    };
//
//
//    template <>
//    struct std::equal<SharedPtr<Object>> {
//        bool operator()(const SharedPtr<Object>& x, const SharedPtr<Object>& y) {
//            return x->equals(y);
//        }
//    };
//}


template <class Interface, class Impl>
class Stub;
