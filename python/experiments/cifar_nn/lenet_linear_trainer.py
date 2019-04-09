import sys
import os
import numpy as np
import time

# sys.path.append("../../../cmake-build-debug/cpp/experiments/cifar_nn")
sys.path.append("../../../cmake-build-release/cpp/experiments/cifar_nn")
import cifar_nn_py

sys.path.append("..")
import data_loading


def main():
    path = os.path.join(os.path.curdir, '../../../resources', 'cifar10')
    train_images, train_labels, test_images, test_labels = data_loading.cifar10(path)
    train_images = train_images.reshape((50000, 3, 32, 32))
    test_images = test_images.reshape((10000, 3, 32, 32))

    # train_images = train_images[:100]
    # train_labels = train_labels[:100]
    # test_images = test_images[:100]
    # test_labels = test_labels[:100]

    ds = cifar_nn_py.PyDataset(train_images, train_labels)

    linear_trainer = cifar_nn_py.PyLetNetLinearTrainer(3, 2, 2)

    start_time = time.time()

    model = linear_trainer.get_trained_model(ds)

    end_time = time.time()
    print('Finished Training in %d sec' % (end_time - start_time))

    res = model.forward(test_images)
    res = np.argmax(res, axis=1)

    accuracy = np.sum((res == test_labels))/res.shape[0]

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * accuracy))


if __name__ == "__main__":
    main()
