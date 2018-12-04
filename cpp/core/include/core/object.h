#pragma once

#include <memory>
#include <variant>

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

};


