#include <core/vec.h>

#include <util/exception.h>

int main(int /*argc*/, char* /*argv*/[]) {
    throw Exception() << "test";
//    std::cout << test() << std::endl;

}
