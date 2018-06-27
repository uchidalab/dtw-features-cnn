import dtw
import time
import numpy as np
import input_data
import network_settings as ns
import math
import csv
import os


def get_dtwfeatures(proto_data, proto_number, local_sample):
    features = np.zeros((50, proto_number))
    for prototype in range(proto_number):
        local_proto = proto_data[prototype]
        output, cost, DTW, path = dtw.dtw(local_proto, local_sample, extended=True)

        for f in range(50):
            features[f, prototype] = cost[path[0][f]][path[1][f]]
    return features


if __name__ == "__main__":

    for version in ["1a", "1b", "1c"]:
        print("Starting: {}".format(version))

        # load settings
        ns.load_settings_raw(version, "1d")
        full_data_file = os.path.join("data", version + "-re-data.txt")
        full_label_file = os.path.join("data", version + "-re-labels.txt")
        # load data
        data_sets = input_data.read_data_sets(full_data_file, full_label_file, ns.IMAGE_SHAPE, test_ratio=0.1, validation_ratio=0.0, pickle=False, boring=False)

        # proto_factor is number of same class prototypes
        proto_factor = 5 if version == "1a" else 2

        no_classes = ns.NUM_CLASSES
        proto_number = proto_factor * no_classes
        # print(proto_number)

        train_data = (data_sets.train.images.reshape((-1, 50, 2)) + 1. ) * (127.5 / 127.)   # this input_data assumes images
        train_labels = data_sets.train.labels

        train_number = np.shape(train_labels)[0]

        test_data = (data_sets.test.images.reshape((-1, 50, 2)) + 1.) * (127.5 / 127.)  # this input_data assumes images
        test_labels = data_sets.test.labels
        test_number = np.shape(test_labels)[0]

        proto_loc = np.zeros(proto_number)

        class_count = np.zeros(no_classes)

        # gets random prototypes with equal class distribution
        for tr in range(train_number):
            cla = int(train_labels[tr])
            if class_count[cla] < proto_factor:
                ind = int((cla * proto_factor) + class_count[cla])
                proto_loc[ind] = tr

                class_count[cla] += 1

        proto_data = train_data[proto_loc]

        # sorts the prototypes for our benefit, no actual effect on anything
        proto_loc[::-1].sort()

        # remove prototypes from training data
        for pl in proto_loc:
            train_data = np.delete(train_data, pl, 0)
            train_labels = np.delete(train_labels, pl, 0)

        # start generation
        test_label_fileloc = os.path.join("data", "test-label-" + version + ".txt")
        test_raw_fileloc = os.path.join("data", "raw-test-data-" + version + ".txt")
        test_dtw_fileloc = os.path.join("data", "dtw_features-50-test-data-" + version + ".txt")
        test_combined_fileloc = os.path.join("data", "dtw_features-50-plus-raw-test-data-" + version + ".txt")
        train_label_fileloc = os.path.join("data", "train-label-" + version + ".txt")
        train_raw_fileloc = os.path.join("data", "raw-train-data-" + version + ".txt")
        train_dtw_fileloc = os.path.join("data", "dtw_features-50-train-data-" + version + ".txt")
        train_combined_fileloc = os.path.join("data", "dtw_features-50-plus-raw-train-data-" + version + ".txt")

        # test set
        with open(test_label_fileloc, 'w') as test_label_file, open(test_raw_fileloc, 'w') as test_raw_file, open(test_dtw_fileloc, 'w') as test_dtw_file, open(test_combined_fileloc, 'w') as test_combined_file:
            writer_test_label = csv.writer(test_label_file, quoting=csv.QUOTE_NONE, delimiter=" ")
            writer_test_raw = csv.writer(test_raw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
            writer_test_dtw = csv.writer(test_dtw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
            writer_test_combined = csv.writer(test_combined_file, quoting=csv.QUOTE_NONE, delimiter=" ")

            for sample in range(test_number):
                local_sample = test_data[sample]
                features = get_dtwfeatures(proto_data, proto_number, local_sample)

                # set the range from 0-255 for the input_data file (the input_data file was made for images and changes it back down to -1 to 1
                features = features * 255.
                local_sample = local_sample * 255.
                class_value = np.argmax(test_labels[sample])

                # write files
                feature_flat = features.reshape(50 * proto_number)
                local_sample_flat = local_sample.reshape(50 * 2)
                writer_test_raw.writerow(local_sample_flat)
                writer_test_dtw.writerow(feature_flat)
                writer_test_combined.writerow(np.append(local_sample_flat, feature_flat))
                writer_test_label.writerow(["{}-{}_test.png".format(class_value, sample), class_value])
        print("{}: Test Done".format(version))

        # train set
        with open(train_label_fileloc, 'w') as train_label_file, open(train_raw_fileloc, 'w') as train_raw_file, open(train_dtw_fileloc, 'w') as train_dtw_file, open(train_combined_fileloc, 'w') as train_combined_file:
            writer_train_label = csv.writer(train_label_file, quoting=csv.QUOTE_NONE, delimiter=" ")
            writer_train_raw = csv.writer(train_raw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
            writer_train_dtw = csv.writer(train_dtw_file, quoting=csv.QUOTE_NONE, delimiter=" ")
            writer_train_combined = csv.writer(train_combined_file, quoting=csv.QUOTE_NONE, delimiter=" ")

            for sample in range(train_number - proto_number):
                local_sample = train_data[sample]
                features = get_dtwfeatures(proto_data, proto_number, local_sample)

                # set the range from 0-255 for the input_data file (the input_data file was made for images and changes it back down to -1 to 1
                features = features * 255.
                local_sample = local_sample * 255.
                class_value = np.argmax(train_labels[sample])

                # write files
                feature_flat = features.reshape(50 * proto_number)
                local_sample_flat = local_sample.reshape(50 * 2)
                writer_train_raw.writerow(local_sample_flat)
                writer_train_dtw.writerow(feature_flat)
                writer_train_combined.writerow(np.append(local_sample_flat, feature_flat))
                writer_train_label.writerow(["{}-{}_train.png".format(class_value, sample), class_value])

                if sample % 1000 == 0:
                    print("{}: Training < {} Done".format(version, str(sample)))
        print("{}: Training Done".format(version))

print("Done")