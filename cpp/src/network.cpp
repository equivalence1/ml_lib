//#include <numeric>
//#include <math.h>
#include <vector>

#include "network.h"




void network::add_to_frame(convolution_t* nd) {
    frame.push_back(nd);
    layers.push_back(new convolution(src_tz_ref, src_memory_ref, src_md_ref, nd, cpu_engine));
    auto cv = dynamic_cast<convolution*>(layers.back());
    src_tz_ref = cv->dst_tz;
    src_md_ref = cv->dst_md;
    src_memory_ref = cv->dst_memory;
    net_fwd.push_back(cv->fwd);
}

void network::add_to_frame(relu_t* nd) {
    frame.push_back(nd);
    layers.push_back(new relu(src_tz_ref, src_memory_ref, src_md_ref, nd, cpu_engine));
    auto cv = dynamic_cast<relu*>(layers.back());
    src_tz_ref = cv->dst_tz;
    src_md_ref = cv->dst_md;
    src_memory_ref = cv->dst_memory;
    net_fwd.push_back(cv->fwd);
}
