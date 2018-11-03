#pragma once
#include "mkldnn.hpp"
#include <numeric>
#include <iostream>
#include <math.h>
//#include <bits/mathcalls.h>

struct node_t {
    virtual ~node_t() = default;
    typedef node_t* type_t;
};

struct convolution_t : node_t {
    convolution_t(int kernel, int stride, int padding, int dims)
            : kernel(kernel)
            , stride(stride)
            , padding(padding)
            , dims(dims)
    {}
    typedef convolution_t* type_t;
    int kernel;
    int stride;
    int padding;
    int dims;
};

struct max_pooling_t : node_t {
    max_pooling_t(int kernel, int stride, int padding, int dims)
            : kernel(kernel)
            , stride(stride)
            , padding(padding)
            , dims(dims)
    {}
    typedef max_pooling_t* type_t;
    int kernel;
    int stride;
    int padding;
    int dims;
};

struct relu_t : node_t {
    const float negative_slope = 1.0f;
    typedef relu_t* type_t;
};


struct node {
    node() = default;
    virtual ~node() = default;
//    virtual void make_forward(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, node_t* nd, mkldnn::engine& cpu_engine);
//    virtual void make_backward();
//    mkldnn::primitive fwd;
//    mkldnn::primitive bwd;
};

struct convolution : node {
//    convolution() = default;
    convolution(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, node_t* nd, mkldnn::engine& cpu_engine)
            : cnv(dynamic_cast<convolution_t*>(nd))
            , conv_weights_tz({cnv->dims, src_tz[1], cnv->kernel, cnv->kernel})
            , conv_bias_tz({cnv->dims})
            , dst_tz({src_tz[0], cnv->dims, (src_tz[2] + cnv->padding*2 + cnv->stride - cnv->kernel)/cnv->stride, (src_tz[2] + cnv->padding*2 + cnv->stride - cnv->kernel)/cnv->stride})
            , conv_strides({cnv->stride, cnv->stride})
            , conv_padding({cnv->padding, cnv->padding})
            , conv_weights(std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1, std::multiplies<uint32_t>()))
            , conv_bias(std::accumulate(conv_bias_tz.begin(), conv_bias_tz.end(), 1, std::multiplies<uint32_t>()))
            , conv_user_weights_memory(mkldnn::memory({ { { conv_weights_tz }, mkldnn::memory::data_type::f32,
                                                          mkldnn::memory::format::oihw },
                                                        cpu_engine },
                                                      conv_weights.data()))
            , conv_user_bias_memory(mkldnn::memory(
                    { { { conv_bias_tz }, mkldnn::memory::data_type::f32, mkldnn::memory::format::x },
                      cpu_engine },
                    conv_bias.data()))
            , conv_weights_md(mkldnn::memory::desc(
                    { conv_weights_tz }, mkldnn::memory::data_type::f32, mkldnn::memory::format::any))
            , conv_bias_md(mkldnn::memory::desc({ conv_bias_tz }, mkldnn::memory::data_type::f32,
                                                mkldnn::memory::format::any))
            , dst_md(mkldnn::memory::desc({ dst_tz }, mkldnn::memory::data_type::f32,
                                          mkldnn::memory::format::any))
            , conv_desc(mkldnn::convolution_forward::desc(
                    mkldnn::prop_kind::forward, mkldnn::convolution_direct, src_md,
                    conv_weights_md, conv_bias_md, dst_md, conv_strides,
                    conv_padding, conv_padding, mkldnn::padding_kind::zero))
            , conv_pd(mkldnn::convolution_forward::primitive_desc(conv_desc, cpu_engine))
            , dst_memory(mkldnn::memory(conv_pd.dst_primitive_desc()))
            , fwd(mkldnn::convolution_forward(conv_pd, src_memory, conv_user_weights_memory,
                                              conv_user_bias_memory, dst_memory))
    {
    }
    ~convolution() = default;
    //forward
    convolution_t* cnv;
    mkldnn::memory::dims conv_weights_tz;
    mkldnn::memory::dims conv_bias_tz;
    mkldnn::memory::dims dst_tz;
    mkldnn::memory::dims conv_strides;
    mkldnn::memory::dims conv_padding;
    std::vector<float> conv_weights;
    std::vector<float> conv_bias;
    mkldnn::memory conv_user_weights_memory;
    mkldnn::memory conv_user_bias_memory;
//
    mkldnn::memory::desc conv_weights_md;
    mkldnn::memory::desc conv_bias_md;
    mkldnn::memory::desc dst_md;
    mkldnn::convolution_forward::desc conv_desc;
    mkldnn::convolution_forward::primitive_desc conv_pd;
    mkldnn::memory dst_memory;
    mkldnn::convolution_forward fwd;
//    std::vector<mkldnn::primitive> fwd;
};

struct relu : node {
//    relu() = default;mkldnn::convolution_forward::desc conv_desc;
    relu(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, node_t* nd, mkldnn::engine& cpu_engine)
            : dst_tz(src_tz)
            , relu_desc(mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                                                      mkldnn::algorithm::eltwise_relu, src_md,
                                                      dynamic_cast<relu_t*>(nd)->negative_slope))
            , relu_pd(mkldnn::eltwise_forward::primitive_desc(relu_desc, cpu_engine))
            , dst_memory(mkldnn::memory(relu_pd.dst_primitive_desc()))
            , dst_md(dst_memory.get_primitive_desc().desc())
            , fwd(mkldnn::eltwise_forward(relu_pd, src_memory, dst_memory))
    {}
    ~relu() = default;
    mkldnn::memory::dims dst_tz;
    mkldnn::eltwise_forward::desc relu_desc;
    mkldnn::eltwise_forward::primitive_desc relu_pd;
    mkldnn::memory dst_memory;
    mkldnn::memory::desc dst_md;
    mkldnn::eltwise_forward fwd;

};


struct network {
    network(int batch_size, int input_channels, int img_size, std::vector<float>& net_src)
            : cpu_engine(mkldnn::engine::cpu, 0)
            , src_tz({batch_size, input_channels, img_size, img_size})
            , src_md(mkldnn::memory::desc({src_tz}, mkldnn::memory::data_type::f32,
                                          mkldnn::memory::format::any))
            , src_memory({{{src_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw}, cpu_engine}, net_src.data())
            , src_tz_ref(src_tz)
            , src_md_ref(src_md)
            , src_memory_ref(src_memory)
    {}
    void add_to_frame(convolution_t* nd);
    void add_to_frame(relu_t* nd);

    std::vector<node*> layers;
    std::vector<node_t*> frame;
    std::vector<mkldnn::primitive> net_fwd;
    mkldnn::engine cpu_engine;
    mkldnn::memory::dims src_tz;
    mkldnn::memory::desc src_md;
    mkldnn::memory src_memory;
    mkldnn::memory::dims& src_tz_ref;
    mkldnn::memory::desc& src_md_ref;
    mkldnn::memory& src_memory_ref;

};
