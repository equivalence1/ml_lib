#include "common.h"
#include <cifar_nn/lenet.h>
#include <cifar_nn/cifar10_reader.h>
#include <cifar_nn/optimizer.h>
#include <cifar_nn/cross_entropy_loss.h>

#include <torch/torch.h>

#include <string>
#include <memory>
#include <iostream>

int main(int argc, char* argv[]) {
    auto device = torch::kCPU;
    if (argc > 1 && std::string(argv[1]) == std::string("CUDA")
            && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA device for training" << std::endl;
    } else {
        std::cout << "Using CPU device for training" << std::endl;
    }

    for (double init_step = 0.001; init_step <  0.01; init_step += 0.001) {
       const int iter = 150;
       for (int reduction = 2; reduction < 11; ++reduction) {
            int x[4][3] = {{20, 40, 80}, {10, 35, 60}, {25, 50, 100}, {30, 60, 90}};
            for(auto threshold: x) {
                        std::cout << init_step << " " << reduction << std::endl;
                        std::cout << threshold[0] << " " << threshold[1] << " " << threshold[2] << std::endl;

			// Init model

			auto lenet = std::make_shared<LeNet>();
			lenet->to(device);

			// Read dataset

			const std::string& path = "../../../../resources/cifar10/cifar-10-batches-bin";
			auto dataset = cifar::read_dataset(path);

			// Create opimizer

			auto optimizer = getDefaultCifar10Optimizer(iter, lenet, device, init_step);
			auto loss = std::make_shared<CrossEntropyLoss>();




			// Attach listeners

			attachDefaultListeners(optimizer, 50000 / 128 / 10, "lenet_checkpoint.pt");

			auto mds = dataset.second.map(getDefaultCifar10TestTransform());
			experiments::Optimizer::emplaceEpochListener<experiments::EpochEndCallback>(optimizer.get(), [&](int epoch, experiments::Model& model) {
				model.eval();

				auto dloader = torch::data::make_data_loader(mds, torch::data::DataLoaderOptions(128));
				int rightAnswersCnt = 0;

				for (auto& batch : *dloader) {
					auto data = batch.data;
					data = data.to(device);
					torch::Tensor target = batch.target;

					torch::Tensor prediction = model.forward(data);
					prediction = torch::argmax(prediction, 1);

					prediction = prediction.to(torch::kCPU);

					auto targetAccessor = target.accessor<int64_t, 1>();
					auto predictionsAccessor = prediction.accessor<int64_t, 1>();
					int size = target.size(0);

					for (int i = 0; i < size; ++i) {
						const int targetClass = targetAccessor[i];
						const int predictionClass = predictionsAccessor[i];
						if (targetClass == predictionClass) {
							rightAnswersCnt++;
						}
					}
				}

				std::cout << "Test accuracy: " <<  rightAnswersCnt * 100.0f / dataset.second.size().value() << std::endl;
			});

			// Train

			optimizer->train(dataset.first, loss, lenet);

			// Eval model

			auto acc = evalModelTestAccEval(dataset.second,
					lenet,
					device,
					getDefaultCifar10TestTransform());

			std::cout << "LeNet test accuracy: " << std::setprecision(2)
					<< acc << "%" << std::endl;
		}
            }
	}
}
