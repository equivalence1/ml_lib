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
using SharedPtr = std::shared_ptr<T>;

//easy to switch for intrusive, if we'll need it
template <class T>
using UniquePtr = std::unique_ptr<T>;

template <class T>
using SharedConstPtr = std::shared_ptr<const T>;

