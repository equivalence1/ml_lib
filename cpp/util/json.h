#pragma once

#include "string.h"

#include <json.hpp>

// for convenience
using json = nlohmann::json;

inline json readJson(const std::string& path) {
    std::ifstream in(path);
    json res;
    in >> res;
    return res;
}

// Json library doesn't support dot notation for nested fields
template<typename T>
inline void setField(json& js, const std::string& field, const T& val) {
    auto path = splitByDelim(field, '.');

    json* j = &js;
    for (int i = 0; i < (int)path.size() - 1; ++i) {
        j = &j->operator[](path[i]);
    }

    j->operator[](*path.rbegin()) = val;
}
