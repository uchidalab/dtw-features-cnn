import os
import numpy as np
from PIL import Image as im
import csv


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    num_classes = int(np.amax(labels_dense) + 1)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, images, labels, one_hot=False):
        #images_2 is series
        assert images.shape[0] == labels.shape[0], ('images_1.shape: %s labels_1.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        # assert images.shape[3] == 1
        #images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
        # Convert from [0, 255] -> [-1.0, 1.0].
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 127.5) - 1.
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images


    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            print("epoch " + str(self._epochs_completed))
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def load_data_from_pickle(data_file, label_file, image_shape):
    import pickle
    print(data_file)
    output = open(data_file, 'rb')
    labels = pickle.load(output)
    images = pickle.load(output)
    output.close()
    images = np.reshape(images, (np.shape(labels)[0], image_shape[0], image_shape[1], image_shape[2]))
    return images, labels

def load_data(data_file, label_file, image_shape, onehot):
    print(data_file)
    images = np.genfromtxt(data_file, delimiter=' ')
    labels = np.genfromtxt(label_file, usecols=(1), delimiter=' ')
    if onehot:
       labels = dense_to_one_hot(labels.astype(int))

    return images, labels

def load_data_from_file(data_file, label_file, image_shape, onehot):
    print(data_file)
    labelsall = np.genfromtxt(label_file, delimiter=' ', dtype=None)
    labelsshape = np.shape(labelsall)
    
    images = np.zeros((labelsshape[0], image_shape[0], image_shape[1], image_shape[2]))
    labels = np.zeros((labelsshape[0]))
    count = 0
    for line in labelsall:
       labels[count] = line[1]
       imagefile = im.open(data_file + line[0].decode("utf-8"))
       imagefile = imagefile.convert('RGB')
       images[count] = np.array(imagefile)
       imagefile.close()
       if count % 1000 == 0:
           print(count)
       count += 1
    if onehot:
       labels = dense_to_one_hot(labels.astype(int))

    return images, labels

def read_data_sets(train_file, train_label, shape, test_file="", test_label="", test_ratio=0.1, validation_ratio=0.0, pickle=True, boring=False, onehot=False):
    class DataSets(object):
        pass

    data_sets = DataSets()

    if (pickle):
        train_images, train_labels = load_data_from_pickle(train_file, train_label, shape)
        if test_file:
            test_images, test_labels = load_data_from_pickle(test_file, test_label, shape)
        else:
            test_size = int(test_ratio * float(train_labels.shape[0]))
            test_images = train_images[:test_size]
            test_labels = train_labels[:test_size]
            train_images = train_images[test_size:]
            train_labels = train_labels[test_size:]
    elif(boring):
        train_images, train_labels = load_data_from_file(train_file, train_label, shape, onehot)
        if test_file:
            test_images, test_labels = load_data_from_file(test_file, test_label, shape, onehot)
        else:
            test_size = int(test_ratio * float(train_labels.shape[0]))
            test_images = train_images[:test_size]
            test_labels = train_labels[:test_size]
            train_images = train_images[test_size:]
            train_labels = train_labels[test_size:]
    else:
        train_images, train_labels = load_data(train_file, train_label, shape, onehot)
        if test_file:
            test_images, test_labels = load_data(test_file, test_label, shape, onehot)
        else:
            test_size = int(test_ratio * float(train_labels.shape[0]))
            test_images = train_images[:test_size]
            test_labels = train_labels[:test_size]
            train_images = train_images[test_size:]
            train_labels = train_labels[test_size:]

    validation_size = int(validation_ratio * float(train_labels.shape[0]))
    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]

    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    
    print("data loaded")
    return data_sets
