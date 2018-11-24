#pragma once
#include<vector>
#include<numeric>
#include "mkldnn.hpp"
#include "network.h"
#include <iostream>


class layer_builder {
public:
    virtual ~layer_builder() = default;
    virtual std::unique_ptr<layer_fwd> build_fwd(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine) = 0;
    virtual std::unique_ptr<layer_bwd> build_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::memory::dims& src_tz, layer_fwd* fwd, mkldnn::engine& cpu_engine) = 0;
};

class convolution_layer_builder
    : public layer_builder {
public:
        convolution_layer_builder(int kernel, int padding, int stride, int out_layers);
        convolution_layer_builder(const convolution_layer_builder& conv);
        std::unique_ptr<layer_fwd> build_fwd(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine);
        std::unique_ptr<layer_bwd> build_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::memory::dims& src_tz, layer_fwd* fwd, mkldnn::engine& cpu_engine);
        int kernel;
        int padding;
        int stride;
        int out_layers;
};

class relu_layer_builder
    : public layer_builder {
public:
        relu_layer_builder() = default;
        relu_layer_builder(const relu_layer_builder& rel);
        std::unique_ptr<layer_fwd> build_fwd(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine);
        std::unique_ptr<layer_bwd> build_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::memory::dims& src_tz, layer_fwd* fwd, mkldnn::engine& cpu_engine);
};

class softmax_layer_builder
        : public layer_builder {
public:
    softmax_layer_builder() = default;
    softmax_layer_builder(const relu_layer_builder& rel);
    std::unique_ptr<layer_fwd> build_fwd(mkldnn::memory::dims& src_tz, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::engine& cpu_engine);
    std::unique_ptr<layer_bwd> build_bwd(mkldnn::memory& diff_dst_memory, mkldnn::memory& src_memory, mkldnn::memory::desc& src_md, mkldnn::memory::dims& src_tz, layer_fwd* fwd, mkldnn::engine& cpu_engine);
};

class network_builder {
public:
    network_builder() = default;
    ~network_builder() = default;
    void add_layer(const convolution_layer_builder& conv);
    void add_layer(const relu_layer_builder& relu);
    void add_layer(const softmax_layer_builder& softm);
    network build(int batch, int dim, int in_chanels);
    std::vector<mkldnn::primitive> forward_net();
    std::vector<mkldnn::primitive> backward_net();
    std::vector<std::unique_ptr<layer_builder>> layers;

};

