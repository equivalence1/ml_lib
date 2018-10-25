"""Load from /home/USER/data/cifar10 or elsewhere; download if missing."""

import tarfile
import os
from urllib.request import urlretrieve
import numpy as np
import sys

sys.path.append("../build")
sys.path.append("../cmake-build-debug")
import nntreepy
from catboost import CatBoostClassifier


def cifar10(path=None):
    r"""Return (train_images, train_labels, test_images, test_labels).

    Args:
        path (str): Directory containing CIFAR-10. Default is
            ./resources/cifar10 or .\resources\cifar10.
            Create if nonexistant. Download CIFAR-10 if missing.

    Returns:
        Tuple of (train_images, train_labels, test_images, test_labels), each
            a matrix. Rows are examples. Columns of images are pixel values,
            with the order (red -> blue -> green). Columns of labels are a
            onehot encoding of the correct class.
    """
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']

    if path is None:
        # Set path to .resources/cifar10
        path = os.path.join(os.path.curdir, 'resources', 'cifar10')

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Download tarfile if missing
    if tar not in os.listdir(path):
        urlretrieve(''.join((url, tar)), os.path.join(path, tar))
        print("Downloaded %s to %s" % (tar, path))

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        # There are 6 files (5 train and 1 test)
        buffr = np.zeros(fsize * 6, dtype='uint8')

        # Get members of tar corresponding to data files
        # -- The tar contains README's and other extraneous stuff
        members = [file for file in tar_object if file.name in files]

        # Sort those members by name
        # -- Ensures we load train data in the proper order
        # -- Ensures that test data is the last file in the list
        members.sort(key=lambda member: member.name)

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    # Split into train and test
    train_images, test_images = images[:50000], images[50000:]
    train_labels, test_labels = labels[:50000], labels[50000:]

    def _onehot(integer_labels):
        """Return matrix whose rows are onehot encodings of integers."""
        n_rows = len(integer_labels)
        n_cols = integer_labels.max() + 1
        onehot = np.zeros((n_rows, n_cols), dtype='uint8')
        onehot[np.arange(n_rows), integer_labels] = 1
        return onehot

    return train_images, _onehot(train_labels), \
           test_images, _onehot(test_labels)


train_images, train_labels, test_images, test_labels = cifar10()

# add one column of 1s in train and test features, so we can use bias
# TODO in future we should do it internally
# ones = np.ones((train_images.shape[0], 1))
# train_images = np.hstack((train_images, ones))
# ones = np.ones((test_images.shape[0], 1))
# test_images = np.hstack((test_images, ones))

ds = nntreepy.DataSet(train_images, train_labels)# np.argmax(train_labels, axis=1).reshape(train_labels.shape[0], 1))

# number_of_samples = 50000
# new_features = ds.test_print(number_of_samples)
# new_features = np.resize(new_features,(number_of_samples, 6 * 6 * 1))
#
# model = CatBoostClassifier(iterations=1000, learning_rate=1, depth=6, loss_function='MultiClass')
# print(np.argmax(train_labels[:number_of_samples],axis=1))
# fit_model = model.fit(new_features, np.argmax(train_labels[:number_of_samples],axis=1))

# print (fit_model.get_params())

#### linear model

# get matrix of weights using least squares
w = nntreepy.least_squares(ds)
print("w: ", w.shape)
print("w[0]:", w[0])
print("w[1]:", w[1])
# print(w)

train_classes = np.argmax(train_labels, axis=1)
print("train_classes:          ", train_classes)
train_classes_predicted = np.argmax(train_images.dot(w), axis=1)
print("train_classes_predicted:", train_classes_predicted)
diff_r = train_classes - train_classes_predicted
sim = np.where(diff_r == 0, 1, 0)
accuracy_rs = np.sum(sim)/sim.shape[0]
print(accuracy_rs)

labels_pred = test_images.dot(w)
classes_pred = np.argmax(labels_pred, axis=1)#np.round(labels_pred)#np.argmax(labels_pred, axis=1) + 1
print(classes_pred.shape)
print(classes_pred)
# print(np.sum(np.where(classes_pred != 2, 1, 0)))
cls_tst = np.argmax(test_labels, axis=1)#.reshape(10000, 1)
print(cls_tst.shape)
diff_r = classes_pred - cls_tst

sim = np.where(diff_r == 0, 1, 0)
accuracy_rs = np.sum(sim)/sim.shape[0]
print(accuracy_rs)

# prediction =
