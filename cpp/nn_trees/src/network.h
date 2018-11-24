#pragma once
#include "mkldnn.hpp"
#include <numeric>
#include <iostream>
#include <math.h>


class layer_fwd {
public:
    virtual ~layer_fwd() = default;

    virtual mkldnn::softmax_forward::primitive_desc* softmax_pd_() = 0;

    virtual mkldnn::eltwise_forward::primitive_desc* relu_pd_() = 0;

    virtual mkldnn::convolution_forward::primitive_desc* conv_pd_() = 0;
    virtual mkldnn::memory* conv_user_weights_memory_() = 0;
    virtual mkldnn::memory::dims* conv_weights_tz_() = 0;
    virtual mkldnn::memory::dims* conv_bias_tz_() = 0;
    virtual mkldnn::memory::dims* conv_strides_() = 0;
    virtual mkldnn::memory::dims* conv_padding_() = 0;

    virtual mkldnn::memory::dims& dst_tz_() = 0;
    virtual mkldnn::memory::desc& dst_md_() = 0;
    virtual mkldnn::memory& dst_memory_() = 0;
    virtual mkldnn::primitive& get_primitive() = 0;
};

class convolution_layer_fwd :
        public layer_fwd {
public:
    convolution_layer_fwd(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine, int kernel, int dims, int padding, int stride);
    ~convolution_layer_fwd() = default;

    mkldnn::softmax_forward::primitive_desc* softmax_pd_();

    mkldnn::eltwise_forward::primitive_desc* relu_pd_();

    mkldnn::convolution_forward::primitive_desc* conv_pd_();
    mkldnn::memory* conv_user_weights_memory_();
    mkldnn::memory::dims* conv_weights_tz_();
    mkldnn::memory::dims* conv_bias_tz_();
    mkldnn::memory::dims* conv_strides_();
    mkldnn::memory::dims* conv_padding_();

    mkldnn::memory::dims& dst_tz_();
    mkldnn::memory::desc& dst_md_();
    mkldnn::memory& dst_memory_();
    mkldnn::primitive& get_primitive();

    mkldnn::memory::dims conv_weights_tz;
    mkldnn::memory::dims conv_bias_tz;
    mkldnn::memory::dims dst_tz;
    mkldnn::memory::dims conv_strides;
    mkldnn::memory::dims conv_padding;
    std::vector<float> conv_weights;
    std::vector<float> conv_bias;
    mkldnn::memory conv_user_weights_memory;
    mkldnn::memory conv_user_bias_memory;

    mkldnn::memory::desc conv_weights_md;
    mkldnn::memory::desc conv_bias_md;
    mkldnn::memory::desc conv_dst_md;
    mkldnn::convolution_forward::desc conv_desc;
    mkldnn::convolution_forward::primitive_desc conv_pd;
    mkldnn::memory::desc dst_md;
    mkldnn::memory dst_memory;
    mkldnn::convolution_forward fwd;


};

class relu_layer_fwd
        : public layer_fwd {
public:

    relu_layer_fwd(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine);
    ~relu_layer_fwd() = default;

    mkldnn::softmax_forward::primitive_desc* softmax_pd_();
    mkldnn::eltwise_forward::primitive_desc* relu_pd_();
    mkldnn::convolution_forward::primitive_desc* conv_pd_();
    mkldnn::memory* conv_user_weights_memory_();
    mkldnn::memory::dims* conv_weights_tz_();
    mkldnn::memory::dims* conv_bias_tz_();
    mkldnn::memory::dims* conv_strides_();
    mkldnn::memory::dims* conv_padding_();

    mkldnn::memory::dims& dst_tz_();
    mkldnn::memory::desc& dst_md_();
    mkldnn::memory& dst_memory_();
    mkldnn::primitive& get_primitive();

    mkldnn::memory::dims dst_tz;
    mkldnn::eltwise_forward::desc relu_desc;
    mkldnn::eltwise_forward::primitive_desc relu_pd;
    mkldnn::memory dst_memory;
    mkldnn::memory::desc dst_md;
    mkldnn::eltwise_forward fwd;
};

class softmax_layer_fwd
        : public layer_fwd {
public:

    softmax_layer_fwd(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine);
    ~softmax_layer_fwd() = default;

    mkldnn::softmax_forward::primitive_desc* softmax_pd_();
    mkldnn::eltwise_forward::primitive_desc* relu_pd_();
    mkldnn::convolution_forward::primitive_desc* conv_pd_();
    mkldnn::memory* conv_user_weights_memory_();
    mkldnn::memory::dims* conv_weights_tz_();
    mkldnn::memory::dims* conv_bias_tz_();
    mkldnn::memory::dims* conv_strides_();
    mkldnn::memory::dims* conv_padding_();

    mkldnn::memory::dims& dst_tz_();
    mkldnn::memory::desc& dst_md_();
    mkldnn::memory& dst_memory_();
    mkldnn::primitive& get_primitive();

    mkldnn::memory::dims dst_tz;
    mkldnn::softmax_forward::desc softmax_desc;
    mkldnn::softmax_forward::primitive_desc softmax_pd;
    mkldnn::memory dst_memory;
    mkldnn::memory::desc dst_md;
    mkldnn::softmax_forward fwd;
};

class layer_bwd {
public:
    virtual ~layer_bwd() = default;
    virtual mkldnn::memory& diff_src_memory_() = 0;
    virtual void push_primitive(std::vector<mkldnn::primitive>& bwd_net) = 0;
};

class convolution_layer_bwd :
        public layer_bwd {
public:
    convolution_layer_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::memory::dims& src_tz, layer_fwd* fwd, mkldnn::engine& cpu_engine);
    mkldnn::memory& diff_src_memory_();
    void push_primitive(std::vector<mkldnn::primitive>& bwd_net);
    std::vector<float> conv_user_diff_weights_buffer;
    std::vector<float> conv_diff_bias_buffer;
    mkldnn::memory conv_user_diff_weights_memory;
    mkldnn::memory conv_diff_bias_memory;
    mkldnn::memory::desc conv_bwd_src_md;
    mkldnn::memory::desc conv_diff_bias_md;
    mkldnn::memory::desc conv_diff_weights_md;
    mkldnn::memory::desc conv_diff_dst_md;
    mkldnn::convolution_backward_weights::desc conv_bwd_weights_desc;
    mkldnn::convolution_backward_weights::primitive_desc conv_bwd_weights_pd;
    mkldnn::convolution_backward_weights bwd_weights;
    mkldnn::convolution_backward_data::desc conv_bwd_desc;
    mkldnn::convolution_backward_data::primitive_desc conv_bwd_pd;
    mkldnn::memory diff_src_memory;
    mkldnn::convolution_backward_data bwd;
};

class relu_layer_bwd :
        public layer_bwd {
public:
    relu_layer_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::memory::dims& src_tz, layer_fwd* fwd, mkldnn::engine& cpu_engine);
    mkldnn::memory& diff_src_memory_();
    void push_primitive(std::vector<mkldnn::primitive>& bwd_net);
    mkldnn::memory::desc diff_dst_md;
    mkldnn::eltwise_backward::desc bwd_desc;
    mkldnn::eltwise_backward::primitive_desc bwd_pd;
    mkldnn::memory diff_src_memory;
    mkldnn::eltwise_backward bwd;
};

class softmax_layer_bwd :
        public layer_bwd {
public:
    softmax_layer_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::memory::dims& src_tz, layer_fwd* fwd, mkldnn::engine& cpu_engine);
    mkldnn::memory& diff_src_memory_();
    void push_primitive(std::vector<mkldnn::primitive>& bwd_net);
    mkldnn::memory::desc diff_dst_md;
    mkldnn::softmax_backward::desc bwd_desc;
    mkldnn::softmax_backward::primitive_desc bwd_pd;
    mkldnn::memory diff_src_memory;
    mkldnn::softmax_backward bwd;
};


class input_data {
public:
    input_data(int batch, int dim, int in_chanels, mkldnn::engine& cpu_engine);
    ~input_data() = default;
    void set_data(std::vector<float> src);
    void set_data(float* data);
    std::vector<float> net_src;
    mkldnn::memory::dims src_tz;
    mkldnn::memory src_memory;
    mkldnn::memory::desc src_md;
};

class output_diff {
public:
    output_diff(mkldnn::memory::dims& dst_tz, mkldnn::engine& cpu_engine);
    ~output_diff() = default;
    void set_diff(std::vector<float> src);
    void set_diff(float* diff);
    std::vector<float> net_diff_dst;
    mkldnn::memory user_diff_dst_memory;
};

class network {
public:
    network();
    void forward_net();
    void backward_net();
    float* operator()(float* data);
    void step(float* diff);
    std::vector<std::unique_ptr<layer_fwd>> fwd_layers;
    std::vector<std::unique_ptr<layer_bwd>> bwd_layers;
    std::unique_ptr<input_data> input_d;
    std::unique_ptr<output_diff> output_df;
    std::vector<mkldnn::primitive> fwd_net;
    std::vector<mkldnn::primitive> bwd_net;
    mkldnn::engine cpu_engine;
};