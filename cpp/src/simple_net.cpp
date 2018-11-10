#include<vector>
#include<numeric>
#include "mkldnn.hpp"
#include "network.h"
#include <iostream>

namespace nntree {
namespace core {

std::vector<float> simple_net(float *data, int batch, int height, int width) {

    int sz = batch * 1 * height *  width;
    std::vector<float> net_src(sz);
    for (int i = 0; i < sz; i++) {
        net_src[i] = data[i];
    }


    auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);

    std::vector<mkldnn::primitive> net;

    convolution_t cnv(3, 1, 0, 8);

    mkldnn::memory::dims src_tz = {batch, 3, 6, 6};
    auto conv_user_src_memory = mkldnn::memory({{{src_tz},
                                                 mkldnn::memory::data_type::f32,
                                                 mkldnn::memory::format::nchw}, cpu_engine}, net_src.data());

    auto conv_src_md = mkldnn::memory::desc({src_tz},
                                            mkldnn::memory::data_type::f32,
                                            mkldnn::memory::format::any);

    convolution cv(src_tz, conv_user_src_memory, conv_src_md, &cnv, cpu_engine);

    relu rel(cv.dst_tz, cv.dst_memory, cv.dst_md, cpu_engine);
    convolution cv2(rel.dst_tz, rel.dst_memory, rel.dst_md, &cnv, cpu_engine);
    relu rel2(cv2.dst_tz, cv2.dst_memory, cv2.dst_md, cpu_engine);


    net.push_back(cv.fwd);
    net.push_back(rel.fwd);
    net.push_back(cv2.fwd);
    net.push_back(rel2.fwd);


    std::vector<float> net_diff_dst(std::accumulate(cv2.dst_tz.begin(), cv2.dst_tz.end(), 1, std::multiplies<uint32_t>()));
    for (size_t i = 0; i < net_diff_dst.size(); ++i)
        net_diff_dst[i] = sinf((float)i);

    auto user_diff_dst_memory = mkldnn::memory({{{cv2.dst_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw },
            cpu_engine}, net_diff_dst.data());
    relu_bwd rel2_bwd(user_diff_dst_memory, cv2.dst_memory, cv2.dst_md, rel2.relu_pd, cpu_engine);
    convolution_bwd cv2_bwd(rel2.dst_tz, rel.dst_md, rel2.dst_memory, cv2, rel2_bwd.diff_src_memory, cpu_engine);
    relu_bwd rel_bwd(cv2_bwd.diff_src_memory, cv.dst_memory, cv.dst_md, rel.relu_pd, cpu_engine);
    convolution_bwd cv_bwd(rel.dst_tz, conv_src_md, rel.dst_memory, cv, rel_bwd.diff_src_memory, cpu_engine);
    std::vector<mkldnn::primitive> bwd_net;
    bwd_net.push_back(rel2_bwd.bwd);
    bwd_net.push_back(cv2_bwd.bwd_weights);
    bwd_net.push_back(cv2_bwd.bwd);
    bwd_net.push_back(rel_bwd.bwd);
    bwd_net.push_back(cv_bwd.bwd_weights);
    bwd_net.push_back(cv_bwd.bwd);

    mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
    mkldnn::stream(mkldnn::stream::kind::eager).submit(bwd_net).wait();

}

}
}
