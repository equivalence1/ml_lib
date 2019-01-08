#include <core/context.h>
#include <gtest/gtest.h>

#include <util/guard.h>


// ops

TEST(SimpleTests, Tests) {
    ComputeDevice gpu(ComputeType::Gpu);
    ComputeDevice cpu(ComputeType::Cpu);
    ComputeDevice current = CurrentDevice();
    on_device(gpu)
        {
            EXPECT_EQ(CurrentDevice(), gpu);
            on_device(cpu)
                {
                    EXPECT_EQ(CurrentDevice(), cpu);
                    on_device(gpu)
                        {
                            EXPECT_EQ(CurrentDevice(), gpu);
                        }
                    EXPECT_EQ(CurrentDevice(), cpu);
                }
            EXPECT_EQ(CurrentDevice(), gpu);
        }
    EXPECT_EQ(CurrentDevice(), current);

}

