#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <string>

inline std::vector<std::string> splitByDelim(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}
