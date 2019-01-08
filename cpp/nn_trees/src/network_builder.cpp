#include "network_builder.h"

convolution_layer_builder::convolution_layer_builder(int kernel, int padding, int stride, int out_layers)
    : kernel(kernel)
      , padding(padding)
      , stride(stride)
      , out_layers(out_layers) {}

convolution_layer_builder::convolution_layer_builder(const convolution_layer_builder& conv)
    : kernel(conv.kernel)
      , padding(conv.padding)
      , stride(conv.stride)
      , out_layers(conv.out_layers) {}

std::unique_ptr<layer_fwd> convolution_layer_builder::build_fwd(
    mkldnn::memory::dims& src_tz,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::engine& cpu_engine) {
    return std::make_unique<convolution_layer_fwd>(src_tz,
                                                   src_memory,
                                                   src_md,
                                                   cpu_engine,
                                                   kernel,
                                                   out_layers,
                                                   padding,
                                                   stride);
}
std::unique_ptr<layer_bwd> convolution_layer_builder::build_bwd(
    mkldnn::memory& diff_dst_memory,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::memory::dims& src_tz,
    layer_fwd* fwd,
    mkldnn::engine& cpu_engine) {
    return std::make_unique<convolution_layer_bwd>(diff_dst_memory, src_memory, src_md, src_tz, fwd, cpu_engine);
}

relu_layer_builder::relu_layer_builder(const relu_layer_builder& rel)
    : relu_layer_builder::relu_layer_builder() {}

std::unique_ptr<layer_fwd> relu_layer_builder::build_fwd(
    mkldnn::memory::dims& src_tz,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::engine& cpu_engine) {
    return std::make_unique<relu_layer_fwd>(src_tz, src_memory, src_md, cpu_engine);
}
std::unique_ptr<layer_bwd> relu_layer_builder::build_bwd(
    mkldnn::memory& diff_dst_memory,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::memory::dims& src_tz,
    layer_fwd* fwd,
    mkldnn::engine& cpu_engine) {
    return std::make_unique<relu_layer_bwd>(diff_dst_memory, src_memory, src_md, src_tz, fwd, cpu_engine);
}

softmax_layer_builder::softmax_layer_builder(const relu_layer_builder& rel)
    : softmax_layer_builder::softmax_layer_builder() {}
std::unique_ptr<layer_fwd> softmax_layer_builder::build_fwd(
    mkldnn::memory::dims& src_tz,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::engine& cpu_engine) {
    return std::make_unique<softmax_layer_fwd>(src_tz, src_memory, src_md, cpu_engine);
}
std::unique_ptr<layer_bwd> softmax_layer_builder::build_bwd(
    mkldnn::memory& diff_dst_memory,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::memory::dims& src_tz,
    layer_fwd* fwd,
    mkldnn::engine& cpu_engine) {
    return std::make_unique<softmax_layer_bwd>(diff_dst_memory, src_memory, src_md, src_tz, fwd, cpu_engine);
}

void network_builder::add_layer(const convolution_layer_builder& conv) {
    layers.push_back(std::make_unique<convolution_layer_builder>(conv));
}

void network_builder::add_layer(const relu_layer_builder& relu) {
    layers.push_back(std::make_unique<relu_layer_builder>(relu));
}

void network_builder::add_layer(const softmax_layer_builder& softm) {
    layers.push_back(std::make_unique<softmax_layer_builder>(softm));
}

network network_builder::build(int batch, int dim, int in_chanels) {//, mkldnn::engine& cpu_engine) {
    network net;
    net.input_d = std::make_unique<input_data>(batch, dim, in_chanels, net.cpu_engine);
    net.fwd_layers.reserve(layers.size());
    net.fwd_layers.push_back(std::move(layers[0]->build_fwd(net.input_d->src_tz,
                                                            net.input_d->src_memory,
                                                            net.input_d->src_md,
                                                            net.cpu_engine)));
    for (uint i = 1; i < layers.size(); ++i) {
        net.fwd_layers.push_back(std::move(layers[i]->build_fwd(net.fwd_layers[i - 1]->dst_tz_(),
                                                                net.fwd_layers[i - 1]->dst_memory_(),
                                                                net.fwd_layers[i - 1]->dst_md_(),
                                                                net.cpu_engine)));
    }
    net.output_df = std::make_unique<output_diff>(net.fwd_layers.back()->dst_tz_(), net.cpu_engine);
    net.bwd_layers.push_back(std::move(layers[layers.size() - 1]->build_bwd(net.output_df->user_diff_dst_memory,
                                                                            net.fwd_layers[layers.size()
                                                                                - 2]->dst_memory_(),
                                                                            net.fwd_layers[layers.size()
                                                                                - 2]->dst_md_(),
                                                                            net.fwd_layers[layers.size()
                                                                                - 2]->dst_tz_(),
                                                                            net.fwd_layers[layers.size() - 1].get(),
                                                                            net.cpu_engine)));
    for (uint64_t i = layers.size() - 2; i > 0; i--) {
        net.bwd_layers.push_back(std::move(layers[i]->build_bwd(net.bwd_layers.back()->diff_src_memory_(),
                                                                net.fwd_layers[i - 1]->dst_memory_(),
                                                                net.fwd_layers[i - 1]->dst_md_(),
                                                                net.fwd_layers[i - 1]->dst_tz_(),
                                                                net.fwd_layers[i].get(),
                                                                net.cpu_engine)));
    }
    net.bwd_layers.push_back(std::move(layers[0]->build_bwd(net.bwd_layers.back()->diff_src_memory_(),
                                                            net.input_d->src_memory,
                                                            net.input_d->src_md,
                                                            net.input_d->src_tz,
                                                            net.fwd_layers[0].get(),
                                                            net.cpu_engine)));
    net.forward_net();
    net.backward_net();
    return net;
}
