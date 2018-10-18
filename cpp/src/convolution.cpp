#include<vector>
#include<numeric>
#include "mkldnn.hpp"
#include <iostream>

std::vector<float> convolution(float *data, int batch, int height, int width) {
  height = 32;
  width = 32;
  std::vector<float> net_src(batch * 3 * height *  width);
  for (int i = 0; i < batch * 3 * height * width; i++) {
    net_src[i] = data[i];
  }

  auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);
  mkldnn::memory::dims conv_src_tz = {batch, 3, height, width};
  mkldnn::memory::dims conv_weights_tz = {1, 3, 11, 11};
  mkldnn::memory::dims conv_bias_tz = {1};
  mkldnn::memory::dims conv_dst_tz = {batch, 1, 6, 6};
  mkldnn::memory::dims conv_strides = {4, 4};
  auto conv_padding = {0, 0};

  std::vector<float> conv_weights(std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1, std::multiplies<uint32_t>()), 1);
  std::vector<float> conv_bias(std::accumulate(conv_bias_tz.begin(), conv_bias_tz.end(), 1, std::multiplies<uint32_t>()), 1);

  auto conv_user_src_memory = mkldnn::memory({{{conv_src_tz},
    mkldnn::memory::data_type::f32,
    mkldnn::memory::format::nchw}, cpu_engine}, net_src.data());

  auto conv_user_weights_memory = mkldnn::memory({{{conv_weights_tz},
    mkldnn::memory::data_type::f32, mkldnn::memory::format::oihw},
    cpu_engine}, conv_weights.data());

  auto conv_user_bias_memory = mkldnn::memory({{{conv_bias_tz},
    mkldnn::memory::data_type::f32, mkldnn::memory::format::x}, cpu_engine},
      conv_bias.data());

  auto conv_src_md = mkldnn::memory::desc({conv_src_tz}, 
    mkldnn::memory::data_type::f32,
    mkldnn::memory::format::any);

  auto conv_bias_md = mkldnn::memory::desc({conv_bias_tz},
    mkldnn::memory::data_type::f32,
    mkldnn::memory::format::any);

  auto conv_weights_md = mkldnn::memory::desc({conv_weights_tz},
    mkldnn::memory::data_type::f32, mkldnn::memory::format::any);

  auto conv_dst_md = mkldnn::memory::desc({conv_dst_tz}, 
    mkldnn::memory::data_type::f32,
    mkldnn::memory::format::any);
  
  //it fails after here so I decide to skip this part

  auto conv_desc = mkldnn::convolution_forward::desc(mkldnn::prop_kind::forward,
    mkldnn::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
    conv_dst_md, conv_strides, conv_padding, conv_padding,
    mkldnn::padding_kind::zero);

  auto conv_prim_desc =	mkldnn::convolution_forward::primitive_desc(conv_desc, cpu_engine);

  std::vector<mkldnn::primitive> net;

  auto conv_src_memory = conv_user_src_memory;
  if (mkldnn::memory::primitive_desc(conv_prim_desc.src_primitive_desc()) !=
  conv_user_src_memory.get_primitive_desc()) {

    conv_src_memory = mkldnn::memory(conv_prim_desc.src_primitive_desc());

    net.push_back(mkldnn::reorder(conv_user_src_memory, conv_src_memory));
  }

  auto conv_weights_memory = conv_user_weights_memory;
  if (mkldnn::memory::primitive_desc(conv_prim_desc.weights_primitive_desc()) !=
      conv_user_weights_memory.get_primitive_desc()) {

    conv_weights_memory = 
      mkldnn::memory(conv_prim_desc.weights_primitive_desc());

    net.push_back(mkldnn::reorder(conv_user_weights_memory, 
      conv_weights_memory));
  }

  auto conv_dst_memory = mkldnn::memory(conv_prim_desc.dst_primitive_desc());

  net.push_back(mkldnn::convolution_forward(conv_prim_desc, conv_src_memory,
    conv_weights_memory, conv_user_bias_memory, conv_dst_memory));
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();

  float* net_output = (float*)conv_dst_memory.get_data_handle();
  std::vector<float> ret(std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1, std::multiplies<uint32_t>()));
  for (size_t i = 0; i != ret.size(); ++i) {
    ret[i] = net_output[i];
  }
  return ret;
}
