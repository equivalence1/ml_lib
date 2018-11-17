#pragma once
#include "mkldnn.hpp"
#include <numeric>
#include <iostream>
#include <math.h>

struct convolution_t {
    convolution_t(int kernel, int stride, int padding, int dims)
            : kernel(kernel)
            , stride(stride)
            , padding(padding)
            , dims(dims)
    {}
    int kernel;
    int stride;
    int padding;
    int dims;
};

struct convolution {
    convolution(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, convolution_t* nd, mkldnn::engine& cpu_engine)
            : cnv(nd)
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
            , conv_dst_md(mkldnn::memory::desc({ dst_tz }, mkldnn::memory::data_type::f32,
                                          mkldnn::memory::format::any))
            , conv_desc(mkldnn::convolution_forward::desc(
                    mkldnn::prop_kind::forward, mkldnn::convolution_direct, src_md,
                    conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
                    conv_padding, conv_padding, mkldnn::padding_kind::zero))
            , conv_pd(mkldnn::convolution_forward::primitive_desc(conv_desc, cpu_engine))
            , dst_md(conv_pd.dst_primitive_desc().desc())
            , dst_memory(mkldnn::memory(conv_pd.dst_primitive_desc()))
            , fwd(mkldnn::convolution_forward(conv_pd, src_memory, conv_user_weights_memory,
                                              conv_user_bias_memory, dst_memory))
    {
    }
    ~convolution() = default;
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
    mkldnn::memory::desc conv_dst_md;
    mkldnn::convolution_forward::desc conv_desc;
    mkldnn::convolution_forward::primitive_desc conv_pd;
    mkldnn::memory::desc dst_md;
    mkldnn::memory dst_memory;
    mkldnn::convolution_forward fwd;
};

struct relu {
    relu(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine)
            : dst_tz(src_tz)
            , relu_desc(mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                                                      mkldnn::algorithm::eltwise_relu, src_md,
                                                      1.0f))
            , relu_pd(mkldnn::eltwise_forward::primitive_desc(relu_desc, cpu_engine))
            , dst_memory(mkldnn::memory(relu_pd.dst_primitive_desc()))
            , dst_md(relu_pd.dst_primitive_desc().desc())
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

struct relu_bwd {
    relu_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::eltwise_forward::primitive_desc& relu_pd, mkldnn::engine& cpu_engine)
        : diff_dst_md(diff_dst_memory.get_primitive_desc().desc())
        , bwd_desc(mkldnn::eltwise_backward::desc(mkldnn::algorithm::eltwise_relu, diff_dst_md, src_md, 1.0f))
        , bwd_pd(mkldnn::eltwise_backward::primitive_desc(bwd_desc, cpu_engine, relu_pd))
        , diff_src_memory(mkldnn::memory(bwd_pd.diff_src_primitive_desc()))
        , bwd(mkldnn::eltwise_backward(bwd_pd, src_memory, diff_dst_memory, diff_src_memory))
    {}
    mkldnn::memory::desc diff_dst_md;
    mkldnn::eltwise_backward::desc bwd_desc;
    mkldnn::eltwise_backward::primitive_desc bwd_pd;
    mkldnn::memory diff_src_memory;
    mkldnn::eltwise_backward bwd;
};

struct convolution_bwd {
    convolution_bwd(mkldnn::memory::dims& src_tz, mkldnn::memory::desc& src_md, mkldnn::memory& src_memory, convolution& conv_fwd, mkldnn::memory& diff_dst_memory, mkldnn::engine& cpu_engine)
        : conv_user_diff_weights_buffer(std::accumulate(conv_fwd.conv_weights_tz.begin(), conv_fwd.conv_weights_tz.end(), 1, std::multiplies<uint32_t>()))
        , conv_diff_bias_buffer(std::accumulate(conv_fwd.conv_bias_tz.begin(), conv_fwd.conv_bias_tz.end(), 1, std::multiplies<uint32_t>()))
        , conv_user_diff_weights_memory(mkldnn::memory({{{conv_fwd.conv_weights_tz}, mkldnn::memory::data_type::f32,
                    mkldnn::memory::format::nchw}, cpu_engine}, conv_user_diff_weights_buffer.data()))
        , conv_diff_bias_memory(mkldnn::memory({{{conv_fwd.conv_bias_tz}, mkldnn::memory::data_type::f32,
                    mkldnn::memory::format::x}, cpu_engine}, conv_diff_bias_buffer.data()))
        , conv_bwd_src_md(src_md)
        , conv_diff_bias_md(mkldnn::memory::desc({conv_fwd.conv_bias_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any))
        , conv_diff_weights_md(mkldnn::memory::desc({conv_fwd.conv_weights_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any))
        , conv_diff_dst_md(mkldnn::memory::desc({conv_fwd.dst_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any))
        , conv_bwd_weights_desc(mkldnn::convolution_backward_weights::desc(mkldnn::convolution_direct, conv_bwd_src_md, conv_diff_weights_md,
            conv_diff_bias_md, conv_diff_dst_md, conv_fwd.conv_strides, conv_fwd.conv_padding, conv_fwd.conv_padding, mkldnn::padding_kind::zero))
        , conv_bwd_weights_pd(mkldnn::convolution_backward_weights::primitive_desc(
                    conv_bwd_weights_desc, cpu_engine, conv_fwd.conv_pd))
        , bwd_weights(mkldnn::convolution_backward_weights(conv_bwd_weights_pd, src_memory, diff_dst_memory,
                            conv_user_diff_weights_memory, conv_diff_bias_memory))
        , conv_bwd_desc(mkldnn::convolution_direct, conv_bwd_src_md, conv_diff_weights_md, conv_diff_dst_md, conv_fwd.conv_strides,
        	 conv_fwd.conv_padding, conv_fwd.conv_padding, mkldnn::padding_kind::zero)
        , conv_bwd_pd(conv_bwd_desc, cpu_engine, conv_fwd.conv_pd)
        , diff_src_memory(conv_bwd_pd.diff_src_primitive_desc())
        , bwd(conv_bwd_pd, diff_dst_memory, conv_fwd.conv_user_weights_memory, diff_src_memory)
    {}
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