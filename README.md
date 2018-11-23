Code conventions:

Build system:  
CMake, https://cliutils.gitlab.io/modern-cmake/

Style guide is based on JMLL

Plus
 
1) c++17 for modern std functionality (filesystem, etc)
2) pragma once for headers
3) pointers near type, not near var name 
4) Filenames: snake_case only, headers *.h; c++ are *.cpp
5) enums are forbidden, enum class instead;


Project structure:

C++:

cpp/:

Public headers: cpp/include/…
Implementation: cpp/src/…
Tests: cpp/tests/…

Python:

python/…


 
Testing framework: Google Test
Python binds: pybind11
