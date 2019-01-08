#include<vector>
#include<numeric>
#include "mkldnn.hpp"
#include "network.h"
#include "network_builder.h"
#include <iostream>

namespace nntree {
    namespace core {

        std::vector<float> simple_net(float* data, int batch, int height, int width) {

            network_builder net_builder;
            net_builder.add_layer(convolution_layer_builder(3, 0, 1, 8));
            net_builder.add_layer(relu_layer_builder());
            net_builder.add_layer(convolution_layer_builder(3, 0, 1, 8));
            net_builder.add_layer(relu_layer_builder());
            auto model = net_builder.build(batch, 6, 3);
//    auto out = model(data);
//    model.step(diff);
        }

    }
}
