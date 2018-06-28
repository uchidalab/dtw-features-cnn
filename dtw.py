__author__ = 'Brian K. Iwana'

import numpy as np
import math


def slow_dtw(base_list, test_list, extended=False):
    """ Computes the DTW of two sequences.
    :param base_list: np array [0..b]
    :param test_list: np array [0..t]
    :param extended: bool
    """

    b = base_list.shape[0]
    t = test_list.shape[0]
    if (b > 0 and t > 0):
        DTW = np.zeros((b, t))
        cost = np.zeros((b, t))

        DTW[:, 0] = float('inf')
        DTW[0, :] = float('inf')
        DTW[0, 0] = 0.0

        for i in range(0, b):
            for j in range(0, t):
                dist = math.sqrt((test_list[j, 0] - base_list[i, 0]) ** 2 + (test_list[j, 1] - base_list[i, 1]) ** 2)
                cost[i, j] = dist
                if (i > 0 and j > 0):
                    jminus2 = DTW[i - 1, j - 2] if j > 1 else float('inf')
                    jminus1 = DTW[i - 1, j - 1]
                    jeven = DTW[i - 1, j]
                    minimum = min(jminus2, jminus1, jeven)
                    DTW[i, j] = dist + minimum
    if (extended):
        return DTW[b - 1, t - 1], cost, DTW, _traceback(DTW)
    else:
        return DTW[b - 1, t - 1]


def fast_dtw(base_list, test_list, extended=False):
    """ Computes the DTW of two sequences.
    :param base_list: np array [0..b]
    :param test_list: np array [0..t]
    :param extended: bool
    """
    b = base_list.shape[0]
    t = test_list.shape[0]
    if (b > 0 and t > 0):
        DTW = np.full((b, t), float('inf'))

        DTW[0, 0] = 0.0
        cost = np.zeros((b, t))
        for i in range(b):
            cost[i] = np.linalg.norm(test_list - base_list[i], axis=1)
        for i in range(1, b):
            DTW[i, 1] = cost[i, 1] + min(DTW[i - 1, 0], DTW[i - 1, 1])
            for j in range(2, t):
                DTW[i, j] = cost[i, j] + min(DTW[i - 1, j - 2], DTW[i - 1, j - 1], DTW[i - 1, j])
    if (extended):
        return DTW[b - 1, t - 1], cost, DTW, _traceback(DTW)
    else:
        return DTW[b - 1, t - 1]


def dtw(base_list, test_list, extended=False, fastdtw=True):
    # fast_dtw is the best and default, but just in case...
    if fastdtw:
        return fast_dtw(base_list, test_list, extended)
    else:
        return slow_dtw(base_list, test_list, extended)


def _traceback(DTW):
    i, j = np.array(DTW.shape) - 1
    p, q = [i], [j]
    while (i > 0 and j > 0):
        tb = np.argmin((DTW[i - 1, j], DTW[i - 1, j - 1], DTW[i - 1, j - 2]))

        if (tb == 0):
            i = i - 1
        elif (tb == 1):
            i = i - 1
            j = j - 1
        elif (tb == 2):
            i = i - 1
            j = j - 2

        p.insert(0, i)
        q.insert(0, j)

    return (np.array(p), np.array(q))


def dtw_draw(cost, DTW, path, train, test):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    # plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    # cost
    plt.subplot(2, 3, 1)
    plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, cost.shape[0] - 0.5))
    plt.ylim((-0.5, cost.shape[0] - 0.5))

    # dtw
    plt.subplot(2, 3, 2)
    plt.imshow(DTW.T, cmap=plt.cm.gray, interpolation='none', origin='lower')
    plt.plot(path[0], path[1], 'y')
    plt.xlim((-0.5, DTW.shape[0] - 0.5))
    plt.ylim((-0.5, DTW.shape[0] - 0.5))

    # training
    plt.subplot(2, 3, 4)
    plt.plot(train[:, 0], train[:, 1], 'b-o')
    plt.xlim((0, 130))
    plt.ylim((0, 130))

    # connection
    plt.subplot(2, 3, 5)
    for i in range(0, path[0].shape[0]):
        plt.plot([train[path[0][i], 0], test[path[1][i], 0]], [train[path[0][i], 1], test[path[1][i], 1]], 'y-')
    plt.plot(test[:, 0], test[:, 1], 'g-o')
    plt.plot(train[:, 0], train[:, 1], 'b-o')
    plt.xlim((0, 130))
    plt.ylim((0, 130))

    # test
    plt.subplot(2, 3, 6)
    plt.plot(test[:, 0], test[:, 1], 'g-o')
    plt.xlim((0, 130))
    plt.ylim((0, 130))

    plt.tight_layout()
    plt.show()
