import dtw
import time
import numpy as np
import input_data
import network_settings as ns
import math
import csv
import os
import sys


def get_dtwfeatures(proto_data, proto_number, local_sample):
    features = np.zeros((50, proto_number))
    for prototype in range(proto_number):
        local_proto = proto_data[prototype]
        output, cost, DTW, path = dtw.dtw(local_proto, local_sample, extended=True)

        for f in range(50):
            features[f, prototype] = cost[path[0][f]][path[1][f]]
    return features

def read_dtw_matrix(version):
    return np.genfromtxt(os.path.join("data", version+"-dtw_matrix.txt"), delimiter=' ')

def random_selection(proto_number):
    # gets random prototypes
    return np.arange(proto_number)

def center_selection(proto_number, distances):
    # gets the center prototypes
    return np.argsort(np.sum(distances, axis=1))[:proto_number]

def border_selection(proto_number, distances):
    # gets the border prototypes
    return np.argsort(np.sum(distances, axis=1))[::-1][:proto_number]

def spanning_selection(proto_number, distances):
    # gets the spanning prototypes
    proto_loc = center_selection(1, distances)
    choice_loc = np.delete(np.arange(np.shape(distances)[0]), proto_loc, 0)
    for iter in range(proto_number-1):
        d = distances[choice_loc]
        p = np.array([choice_loc[np.argmax(np.min(d[:,proto_loc], axis=1))]])
        proto_loc = np.append(proto_loc, p)
        choice_loc = np.delete(choice_loc, p, 0)
    return proto_loc

def k_centers_selection(proto_number, distances):
    # finds k centers
    no_possible = np.shape(distances)[0]

    # initialize with spanning
    proto_loc = spanning_selection(proto_number, distances)
    for iter in range(1000):
        # assign every point into a group with the centers
        membership = np.zeros(no_possible, dtype=np.int32)
        for i, d in enumerate(distances):
            membership[i] = proto_loc[np.argmin(d[proto_loc])]

        # find center of groups
        was_change = False
        for i, p in enumerate(proto_loc):
            p_group = np.where(membership==p)[0]
            new_center = p_group[center_selection(1, distances[p_group])][0]
            if new_center != p:
                proto_loc[i] = new_center
                was_change = True
        if was_change == False:
            print("stopping at {}".format(iter))
            break
    return proto_loc

def k_medoids_selection(proto_number, distances):
    # A simple and fast algorithm for K-medoids clustering
    no_possible = np.shape(distances)[0]

    # initialize using v_j = \sum^n_{i=1}{\frac{d_{ij}{\sum^n_{l=1}{d_{il}}}}
    sums = np.sum(distances, axis=1)
    ratings = np.zeros(no_possible)
    for c in np.arange(no_possible):
        ratings[c] = np.sum(distances[c] / sums)
    proto_loc = np.argsort(ratings)[:proto_number]

    for iter in range(1000):
        # assign every point into a group with the centers
        membership = np.zeros(no_possible, dtype=np.int32)
        for i, d in enumerate(distances):
            membership[i] = proto_loc[np.argmin(d[proto_loc])]
        # find center of groups
        proto_sum = np.sum(distances[proto_loc])
        for i, p in enumerate(proto_loc):
            p_group = np.where(membership==p)[0]
            proto_loc[i] = p_group[center_selection(1, distances[p_group])][0]
        if proto_sum == np.sum(distances[proto_loc]):
            print("stopping at {}".format(iter))
            break
    return proto_loc

def selector_selector(selection, proto_number, distances):
    if selection == "random":
        return random_selection(proto_number)
    elif selection == "centers":
        return center_selection(proto_number, distances)
    elif selection == "borders":
        return border_selection(proto_number, distances)
    elif selection == "spanning":
        return spanning_selection(proto_number, distances)
    elif selection == "kmedoids":
        return k_medoids_selection(proto_number, distances)
    elif selection == "kcenters":
        return k_centers_selection(proto_number, distances)
    else:
        return random_selection(proto_number)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Error, Syntax: {0} [version] [prototype selection] [classwise/independent] [prototype number]".format(sys.argv[0]))
        exit()
    version = sys.argv[1]
    selection = sys.argv[2]
    classwise = sys.argv[3]
    proto_number = int(sys.argv[4])

    print("Starting: {}".format(version))

    # load settings
    ns.load_settings_raw(version, "1d")
    full_data_file = os.path.join("data", version + "-re-data.txt")
    full_label_file = os.path.join("data", version + "-re-labels.txt")
    # load data
    data_sets = input_data.read_data_sets(full_data_file, full_label_file, ns.IMAGE_SHAPE, test_ratio=0.1,
                                          validation_ratio=0.0, pickle=False, boring=False)

    no_classes = ns.NUM_CLASSES
    # print(proto_number)

    train_data = (data_sets.train.images.reshape((-1, 50, 2)) + 1.) * (
            127.5 / 127.)  # this input_data assumes images
    train_labels = data_sets.train.labels[:200]

    train_number = np.shape(train_labels)[0]

    test_data = (data_sets.test.images.reshape((-1, 50, 2)) + 1.) * (127.5 / 127.)  # this input_data assumes images
    test_labels = data_sets.test.labels
    test_number = np.shape(test_labels)[0]

    distances = read_dtw_matrix(version)

    if classwise == "classwise":
        proto_loc = np.zeros(0, dtype=np.int32)
        proto_factor = int(proto_number / no_classes)
        for c in range(no_classes):
            cw = np.where(train_labels == c)[0]
            cw_distances = distances[cw]
            cw_distances = cw_distances[:,cw]
            cw_proto = selector_selector(selection, proto_factor, cw_distances)
            proto_loc = np.append(proto_loc, cw[cw_proto])
    else:
        proto_loc = selector_selector(selection, proto_number, distances)

    proto_data = train_data[proto_loc]
    print(proto_loc)
    exit()
    print("Selection Done.")

    # sorts the prototypes so deletion happens in reverse order and doesn't interfere with indices
    proto_loc[::-1].sort()

    # remove prototypes from training data
    for pl in proto_loc:
        train_data = np.delete(train_data, pl, 0)
        train_labels = np.delete(train_labels, pl, 0)

    # start generation
    test_label_fileloc = os.path.join("data", "test-label-{}-{}-{}.txt".format(version, selection, proto_number))
    test_raw_fileloc = os.path.join("data", "raw-test-data-{}-{}-{}.txt".format(version, selection, proto_number))
    test_dtw_fileloc = os.path.join("data",
                                    "dtw_features-test-data-{}-{}-{}.txt".format(version, selection, proto_number))
    test_combined_fileloc = os.path.join("data",
                                         "dtw_features-plus-raw-test-data-{}-{}-{}.txt".format(version, selection,
                                                                                               proto_number))
    train_label_fileloc = os.path.join("data", "train-label-{}-{}-{}.txt".format(version, selection, proto_number))
    train_raw_fileloc = os.path.join("data", "raw-train-data-{}-{}-{}.txt".format(version, selection, proto_number))
    train_dtw_fileloc = os.path.join("data", "dtw_features-train-data-{}-{}-{}.txt".format(version, selection,
                                                                                           proto_number))
    train_combined_fileloc = os.path.join("data",
                                          "dtw_features-plus-raw-train-data-{}-{}-{}.txt".format(version, selection,
                                                                                                 proto_number))

    # test set
    with open(test_label_fileloc, 'w') as test_label_file, open(test_raw_fileloc, 'w') as test_raw_file, open(
            test_dtw_fileloc, 'w') as test_dtw_file, open(test_combined_fileloc, 'w') as test_combined_file:
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
    with open(train_label_fileloc, 'w') as train_label_file, open(train_raw_fileloc, 'w') as train_raw_file, open(
            train_dtw_fileloc, 'w') as train_dtw_file, open(train_combined_fileloc, 'w') as train_combined_file:
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