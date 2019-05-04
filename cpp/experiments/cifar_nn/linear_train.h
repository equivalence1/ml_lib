//#pragma once
//
//#include "em_like_train.h"
//#include "lenet.h"
//#include "linear_model.h"
//#include "cross_entropy_loss.h"
//
//#include <torch/torch.h>
//
//class LeNetLinearTrainer : public EMLikeTrainer<torch::data::transforms::Stack<>> {
//public:
//    LeNetLinearTrainer(uint32_t it_global,
//            uint32_t it_repr,
//            uint32_t it_decision) : EMLikeTrainer(torch::data::transforms::Stack<>(), it_global, model) {
//        model_->conv().reset(new LeNetConv());
//        model_->conv()->to(torch::kCUDA);
//
//        // TODO optimizers
//
////        representationOptimizer_ = std::make_shared<DefaultSGDOptimizer>(it_repr, reprOptimOptions);
////        {
////            torch::optim::SGDOptions reprOptimOptions(0.001);
////            auto transform = torch::data::transforms::Stack<>();
////            experiments::OptimizerArgs<decltype(transform)> args(transform, it_repr, torch::kCUDA);
////
////            auto dloaderOptions = torch::data::DataLoaderOptions(4);
////            args.dloaderOptions_ = std::move(dloaderOptions);
////
////            auto optim = std::make_shared<torch::optim::SGD>(representationsModel_->parameters(), reprOptimOptions);
////            args.torchOptim_ = optim;
////
////            auto lr = &(optim->options.learning_rate_);
////            args.lrPtrGetter_ = [=]() { return lr; };
////
////            auto optimizer = std::make_shared<experiments::DefaultOptimizer<decltype(args.transform_)>>(args);
//////            attachDefaultListeners(optimizer, 50000 / 4 / 10, "lenet_em_conv_checkpoint.pt");
////            representationOptimizer_ = optimizer;
////        }
////
////        decisionModel_ = std::make_shared<LinearModel>(16 * 5 * 5, 10);
////        decisionModel_->to(torch::kCUDA);
////
////        {
////            torch::optim::SGDOptions decisionOptimOptions(0.1);
////            auto transform = torch::data::transforms::Stack<>();
////            experiments::OptimizerArgs<decltype(transform)> args(transform, it_decision, torch::kCUDA);
////
////            auto dloaderOptions = torch::data::DataLoaderOptions(4);
////            args.dloaderOptions_ = std::move(dloaderOptions);
////
////            auto optim = std::make_shared<torch::optim::SGD>(decisionModel_->parameters(), decisionOptimOptions);
////            args.torchOptim_ = optim;
////
////            auto lr = &(optim->options.learning_rate_);
////            args.lrPtrGetter_ = [=]() { return lr; };
////
////            auto optimizer = std::make_shared<experiments::DefaultOptimizer<decltype(args.transform_)>>(args);
//////            attachDefaultListeners(optimizer, 50000 / 4 / 10, "lenet_em_conv_checkpoint.pt");
////            decisionFuncOptimizer_ = optimizer;
////        }
////        decisionFuncOptimizer_ = std::make_shared<DefaultSGDOptimizer>(it_decision, decisionOptimOptions);
//
//    }
//
//    experiments::ModelPtr getTrainedModel(TensorPairDataset& ds) {
//        LossPtr loss = std::make_shared<CrossEntropyLoss>();
//        return EMLikeTrainer::getTrainedModel(ds, loss);
//    }
//
//};
