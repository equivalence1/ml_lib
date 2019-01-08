#include <vector>
#include "network.h"

convolution_layer_fwd::convolution_layer_fwd(
    mkldnn::memory::dims& src_tz,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::engine& cpu_engine,
    int kernel,
    int dims,
    int padding,
    int stride)
    : conv_weights_tz({dims, src_tz[1], kernel, kernel})
      , conv_bias_tz({dims})
      , dst_tz({src_tz[0], dims, (src_tz[2] + padding * 2 + stride - kernel) / stride,
                (src_tz[2] + padding * 2 + stride - kernel) / stride})
      , conv_strides({stride, stride})
      , conv_padding({padding, padding})
      , conv_weights(std::accumulate(conv_weights_tz.begin(), conv_weights_tz.end(), 1, std::multiplies<uint32_t>()),
                     1.)
      , conv_bias(std::accumulate(conv_bias_tz.begin(), conv_bias_tz.end(), 1, std::multiplies<uint32_t>()), 0.)
      , conv_user_weights_memory(mkldnn::memory({{{conv_weights_tz}, mkldnn::memory::data_type::f32,
                                                  mkldnn::memory::format::oihw},
                                                 cpu_engine},
                                                conv_weights.data()))
      , conv_user_bias_memory(mkldnn::memory(
        {{{conv_bias_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::x},
         cpu_engine},
        conv_bias.data()))
      , conv_weights_md(mkldnn::memory::desc(
        {conv_weights_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::any))
      , conv_bias_md(mkldnn::memory::desc({conv_bias_tz}, mkldnn::memory::data_type::f32,
                                          mkldnn::memory::format::any))
      , conv_dst_md(mkldnn::memory::desc({dst_tz}, mkldnn::memory::data_type::f32,
                                         mkldnn::memory::format::any))
      , conv_desc(mkldnn::convolution_forward::desc(
        mkldnn::prop_kind::forward, mkldnn::convolution_direct, src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
        conv_padding, conv_padding, mkldnn::padding_kind::zero))
      , conv_pd(mkldnn::convolution_forward::primitive_desc(conv_desc, cpu_engine))
      , dst_md(conv_pd.dst_primitive_desc().desc())
      , dst_memory(mkldnn::memory(conv_pd.dst_primitive_desc()))
      , fwd(mkldnn::convolution_forward(conv_pd, src_memory, conv_user_weights_memory,
                                        conv_user_bias_memory, dst_memory)) {}

mkldnn::softmax_forward::primitive_desc* convolution_layer_fwd::softmax_pd_() { return nullptr; }
mkldnn::eltwise_forward::primitive_desc* convolution_layer_fwd::relu_pd_() { return nullptr; }

mkldnn::convolution_forward::primitive_desc* convolution_layer_fwd::conv_pd_() { return &conv_pd; }
mkldnn::memory* convolution_layer_fwd::conv_user_weights_memory_() { return &conv_user_weights_memory; }
mkldnn::memory::dims* convolution_layer_fwd::conv_weights_tz_() { return &conv_weights_tz; }
mkldnn::memory::dims* convolution_layer_fwd::conv_bias_tz_() { return &conv_bias_tz; }
mkldnn::memory::dims* convolution_layer_fwd::conv_strides_() { return &conv_strides; }
mkldnn::memory::dims* convolution_layer_fwd::conv_padding_() { return &conv_padding; }

mkldnn::memory::dims& convolution_layer_fwd::dst_tz_() { return dst_tz; }
mkldnn::memory::desc& convolution_layer_fwd::dst_md_() { return dst_md; }
mkldnn::memory& convolution_layer_fwd::dst_memory_() { return dst_memory; }
mkldnn::primitive& convolution_layer_fwd::get_primitive() { return fwd; }

relu_layer_fwd::relu_layer_fwd(
    mkldnn::memory::dims& src_tz,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::engine& cpu_engine)
    : dst_tz(src_tz)
      , relu_desc(mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward,
                                                mkldnn::algorithm::eltwise_relu, src_md,
                                                1.0f))
      , relu_pd(mkldnn::eltwise_forward::primitive_desc(relu_desc, cpu_engine))
      , dst_memory(mkldnn::memory(relu_pd.dst_primitive_desc()))
      , dst_md(relu_pd.dst_primitive_desc().desc())
      , fwd(mkldnn::eltwise_forward(relu_pd, src_memory, dst_memory)) {}

mkldnn::softmax_forward::primitive_desc* relu_layer_fwd::softmax_pd_() { return nullptr; }
mkldnn::eltwise_forward::primitive_desc* relu_layer_fwd::relu_pd_() { return &relu_pd; }
mkldnn::convolution_forward::primitive_desc* relu_layer_fwd::conv_pd_() { return nullptr; }
mkldnn::memory* relu_layer_fwd::conv_user_weights_memory_() { return nullptr; }
mkldnn::memory::dims* relu_layer_fwd::conv_weights_tz_() { return nullptr; }
mkldnn::memory::dims* relu_layer_fwd::conv_bias_tz_() { return nullptr; }
mkldnn::memory::dims* relu_layer_fwd::conv_strides_() { return nullptr; }
mkldnn::memory::dims* relu_layer_fwd::conv_padding_() { return nullptr; }

mkldnn::memory::dims& relu_layer_fwd::dst_tz_() { return dst_tz; }
mkldnn::memory::desc& relu_layer_fwd::dst_md_() { return dst_md; }
mkldnn::memory& relu_layer_fwd::dst_memory_() { return dst_memory; }
mkldnn::primitive& relu_layer_fwd::get_primitive() { return fwd; }

softmax_layer_fwd::softmax_layer_fwd(
    mkldnn::memory::dims& src_tz,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::engine& cpu_engine)
    : dst_tz(src_tz)
      , softmax_desc(mkldnn::softmax_forward::desc(mkldnn::prop_kind::forward, src_md, 2))//softmax_axis
      , softmax_pd(mkldnn::softmax_forward::primitive_desc(softmax_desc, cpu_engine))
      , dst_memory(mkldnn::memory(src_memory.get_primitive_desc()))
      , dst_md(src_memory.get_primitive_desc().desc())
      , fwd(mkldnn::softmax_forward(softmax_pd, src_memory, dst_memory)) {}

mkldnn::softmax_forward::primitive_desc* softmax_layer_fwd::softmax_pd_() { return &softmax_pd; }
mkldnn::eltwise_forward::primitive_desc* softmax_layer_fwd::relu_pd_() { return nullptr; }
mkldnn::convolution_forward::primitive_desc* softmax_layer_fwd::conv_pd_() { return nullptr; }
mkldnn::memory* softmax_layer_fwd::conv_user_weights_memory_() { return nullptr; }
mkldnn::memory::dims* softmax_layer_fwd::conv_weights_tz_() { return nullptr; }
mkldnn::memory::dims* softmax_layer_fwd::conv_bias_tz_() { return nullptr; }
mkldnn::memory::dims* softmax_layer_fwd::conv_strides_() { return nullptr; }
mkldnn::memory::dims* softmax_layer_fwd::conv_padding_() { return nullptr; }

mkldnn::memory::dims& softmax_layer_fwd::dst_tz_() { return dst_tz; }
mkldnn::memory::desc& softmax_layer_fwd::dst_md_() { return dst_md; }
mkldnn::memory& softmax_layer_fwd::dst_memory_() { return dst_memory; }
mkldnn::primitive& softmax_layer_fwd::get_primitive() { return fwd; }

relu_layer_bwd::relu_layer_bwd(
    mkldnn::memory& diff_dst_memory,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::memory::dims& src_tz,
    layer_fwd* fwd,
    mkldnn::engine& cpu_engine)
    : diff_dst_md(diff_dst_memory.get_primitive_desc().desc())
      , bwd_desc(mkldnn::eltwise_backward::desc(mkldnn::algorithm::eltwise_relu, diff_dst_md, src_md, 1.0f))
      , bwd_pd(mkldnn::eltwise_backward::primitive_desc(bwd_desc, cpu_engine, *fwd->relu_pd_()))
      , diff_src_memory(mkldnn::memory(bwd_pd.diff_src_primitive_desc()))
      , bwd(mkldnn::eltwise_backward(bwd_pd, src_memory, diff_dst_memory, diff_src_memory)) {}
mkldnn::memory& relu_layer_bwd::diff_src_memory_() { return diff_src_memory; }
void relu_layer_bwd::push_primitive(std::vector<mkldnn::primitive>& bwd_net) {
    bwd_net.push_back(bwd);
}

softmax_layer_bwd::softmax_layer_bwd(
    mkldnn::memory& diff_dst_memory,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::memory::dims& src_tz,
    layer_fwd* fwd,
    mkldnn::engine& cpu_engine)
    : diff_dst_md(diff_dst_memory.get_primitive_desc().desc())
      , bwd_desc(mkldnn::softmax_backward::desc(diff_dst_md, src_md, 2))
      , bwd_pd(mkldnn::softmax_backward::primitive_desc(bwd_desc, cpu_engine, *fwd->softmax_pd_()))
      , diff_src_memory(mkldnn::memory(bwd_pd.diff_src_primitive_desc()))
      , bwd(mkldnn::softmax_backward(bwd_pd, src_memory, diff_dst_memory, diff_src_memory)) {}
mkldnn::memory& softmax_layer_bwd::diff_src_memory_() { return diff_src_memory; }
void softmax_layer_bwd::push_primitive(std::vector<mkldnn::primitive>& bwd_net) {
    bwd_net.push_back(bwd);
}

convolution_layer_bwd::convolution_layer_bwd(
    mkldnn::memory& diff_dst_memory,
    mkldnn::memory& src_memory,
    mkldnn::memory::desc& src_md,
    mkldnn::memory::dims& src_tz,
    layer_fwd* fwd,
    mkldnn::engine& cpu_engine)
    : conv_user_diff_weights_buffer(std::accumulate(fwd->conv_weights_tz_()->begin(),
                                                    fwd->conv_weights_tz_()->end(),
                                                    1,
                                                    std::multiplies<uint32_t>()), 0.)
      , conv_diff_bias_buffer(std::accumulate(fwd->conv_bias_tz_()->begin(),
                                              fwd->conv_bias_tz_()->end(),
                                              1,
                                              std::multiplies<uint32_t>()), 0.)
      , conv_user_diff_weights_memory(mkldnn::memory({{{*fwd->conv_weights_tz_()}, mkldnn::memory::data_type::f32,
                                                       mkldnn::memory::format::nchw}, cpu_engine},
                                                     conv_user_diff_weights_buffer.data()))
      , conv_diff_bias_memory(mkldnn::memory({{{*fwd->conv_bias_tz_()}, mkldnn::memory::data_type::f32,
                                               mkldnn::memory::format::x}, cpu_engine}, conv_diff_bias_buffer.data()))
      , conv_bwd_src_md(src_md)
      , conv_diff_bias_md(mkldnn::memory::desc({*fwd->conv_bias_tz_()},
                                               mkldnn::memory::data_type::f32,
                                               mkldnn::memory::format::any))
      , conv_diff_weights_md(mkldnn::memory::desc({*fwd->conv_weights_tz_()},
                                                  mkldnn::memory::data_type::f32,
                                                  mkldnn::memory::format::any))
      , conv_diff_dst_md(mkldnn::memory::desc({fwd->dst_tz_()},
                                              mkldnn::memory::data_type::f32,
                                              mkldnn::memory::format::any))
      , conv_bwd_weights_desc(mkldnn::convolution_backward_weights::desc(mkldnn::convolution_direct,
                                                                         conv_bwd_src_md,
                                                                         conv_diff_weights_md,
                                                                         conv_diff_bias_md,
                                                                         conv_diff_dst_md,
                                                                         *fwd->conv_strides_(),
                                                                         *fwd->conv_padding_(),
                                                                         *fwd->conv_padding_(),
                                                                         mkldnn::padding_kind::zero))
      , conv_bwd_weights_pd(mkldnn::convolution_backward_weights::primitive_desc(
        conv_bwd_weights_desc, cpu_engine, *fwd->conv_pd_()))
      , bwd_weights(mkldnn::convolution_backward_weights(conv_bwd_weights_pd, src_memory, diff_dst_memory,
                                                         conv_user_diff_weights_memory, conv_diff_bias_memory))
      , conv_bwd_desc(mkldnn::convolution_direct,
                      conv_bwd_src_md,
                      conv_diff_weights_md,
                      conv_diff_dst_md,
                      *fwd->conv_strides_(),
                      *fwd->conv_padding_(),
                      *fwd->conv_padding_(),
                      mkldnn::padding_kind::zero)
      , conv_bwd_pd(conv_bwd_desc, cpu_engine, *fwd->conv_pd_())
      , diff_src_memory(conv_bwd_pd.diff_src_primitive_desc())
      , bwd(conv_bwd_pd, diff_dst_memory, *fwd->conv_user_weights_memory_(), diff_src_memory) {}
mkldnn::memory& convolution_layer_bwd::diff_src_memory_() { return diff_src_memory; }
void convolution_layer_bwd::push_primitive(std::vector<mkldnn::primitive>& bwd_net) {
    bwd_net.push_back(bwd_weights);
    bwd_net.push_back(bwd);
}

input_data::input_data(int batch, int dim, int in_chanels, mkldnn::engine& cpu_engine)
    : net_src(batch * in_chanels * dim * dim)
      , src_tz({batch, in_chanels, dim, dim})
      , src_memory(mkldnn::memory({{{src_tz},
                                    mkldnn::memory::data_type::f32,
                                    mkldnn::memory::format::nchw}, cpu_engine}, net_src.data()))

      , src_md(mkldnn::memory::desc({src_tz},
                                    mkldnn::memory::data_type::f32,
                                    mkldnn::memory::format::any)) {}

void input_data::set_data(std::vector<float> src) { net_src = std::move(src); }
void input_data::set_data(float* data) { net_src.assign(data, data + net_src.size()); }

output_diff::output_diff(mkldnn::memory::dims& dst_tz, mkldnn::engine& cpu_engine)
    : net_diff_dst(std::accumulate(dst_tz.begin(), dst_tz.end(), 1, std::multiplies<uint32_t>()))
      , user_diff_dst_memory(mkldnn::memory({{{dst_tz}, mkldnn::memory::data_type::f32, mkldnn::memory::format::nchw},
                                             cpu_engine}, net_diff_dst.data())) {
    for (size_t i = 0; i < net_diff_dst.size(); ++i)
        net_diff_dst[i] = sinf((float) i);
}
void output_diff::set_diff(std::vector<float> diff) { net_diff_dst = diff; }
void output_diff::set_diff(float* diff) { net_diff_dst.assign(diff, diff + net_diff_dst.size()); }

network::network()
    : cpu_engine(mkldnn::engine(mkldnn::engine::cpu, 0)) {}

void network::forward_net() {
    for (uint i = 0; i < fwd_layers.size(); ++i) {
        fwd_net.push_back(fwd_layers[i]->get_primitive());
    }
}

void network::backward_net() {
    for (uint i = 0; i < bwd_layers.size(); ++i) {
        bwd_layers[i]->push_primitive(bwd_net);
    }
}

float* network::operator()(float* input) {
    input_d->set_data(input);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(fwd_net).wait();
    return static_cast<float*>(fwd_layers.back()->dst_memory_().get_data_handle());
}

void network::step(float* diff) {
    output_df->set_diff(diff);
    mkldnn::stream(mkldnn::stream::kind::eager).submit(bwd_net).wait();
    //optimizer step to update weights
}
