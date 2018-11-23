#include <gtest/gtest.h>
#include <core/vec.h>
#include <core/vec_factory.h>
#include <core/vec_tools/fill.h>
#include <core/funcs/linear.h>
#include <util/exception.h>

TEST(FuncTests, Linear) {
    Vec param = VecFactory::create(VecType::Cpu, 3);

    param.set(0, 1);
    param.set(1, -2);
    param.set(2, 3);


    Vec x = VecFactory::create(VecType::Cpu, 2);
    x.set(0, 10);
    x.set(1, 20);

    Linear linear(param);
    double res = linear.value(x);
    EXPECT_EQ(res, 41);

}


