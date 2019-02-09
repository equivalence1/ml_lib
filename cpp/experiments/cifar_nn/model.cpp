#include "model.h"

void train_model(Model* model, TensorPairDataset* d, int epochs) {
    auto mds = d->map(torch::data::transforms::Stack<>());
    auto dloader = torch::data::make_data_loader(mds, 64);

    torch::optim::SGD optimizer(model->parameters(), 0.01);

    for (int epoch = 0; epoch < epochs; epoch++) {
        int batch_index = 0;
        for (auto& batch : *dloader) {
            optimizer.zero_grad();
            auto prediction = model->forward(batch.data);

            // TODO(equivalence1) log_softmax works better then log(softmax)
            prediction = torch::log(prediction);
            auto loss= torch::nll_loss(prediction, batch.target);
            loss.backward();
            optimizer.step();

            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                          << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
//                            torch::save(this, "net.pt");
            }
        }
    }
}
