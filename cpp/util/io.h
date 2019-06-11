#pragma once

// include json-related io
#include "json.h"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

inline std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream strStream;
    strStream << in.rdbuf(); //read the file
    std::string params = strStream.str();
    return params;
}
