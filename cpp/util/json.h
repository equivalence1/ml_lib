#pragma once

#include <json.hpp>

// for convenience
using json = nlohmann::json;

inline json readJson(const std::string& path) {
    std::ifstream in(path);
    json res;
    in >> res;
    return res;
}
