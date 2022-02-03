
import sys
import numpy as np

NUM_NEURONS_IN_LAYER = 100
NUM_NEURONS_IN_LAST_LAYER = 10
NUM_PIXELS_OF_PICTURE = 784
RATE = 0.3
EPOCHS = 12

# good seed for initialize the data by shuffling
FIRST_SEED = 967
# good seeds for shuffle in eac epoch
SEEDS_ARRAY = [722, 766, 54, 185, 927, 36, 604, 909, 108, 782, 885, 128]
# for checking
PARTS = 1
K_FOLD = 20

sigmoid = lambda x: 1 / (1 + np.exp(-x))


def softmax(arr):
    """
    activation function
    :param arr: the vector that needed ro come inside the activation
    :return: vector after the activation calculated
    """
    arr = arr - max(arr)
    sum = np.sum(np.exp(arr))
    return np.exp(arr) / sum


def fprop(x, y, params):
    """
    forward propagation. find the y_hats and calculate the vectors and matrixs for the next layer.
    :param x: the picture
    :param y: the real label
    :param params: the parameters - wights and bias
    :return: all the information that was clculated in this function
    """
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1

    z1 /= NUM_NEURONS_IN_LAYER
    h1 = sigmoid(z1)

    z2 = np.dot(W2, h1) + b2
    z2 /= NUM_NEURONS_IN_LAST_LAYER

    h2 = softmax(z2)

    y_indexes = np.zeros(NUM_NEURONS_IN_LAST_LAYER)
    y_indexes[int(y)] = 1
    y_indexes = y_indexes[:, None]

    # loss = -np.log(h2)
    ret = {'x': x, 'y': y_indexes, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(fprop_cache):
    """
    backward propagation. updating the weights and bias according to their derivatives.
    :param fprop_cache: the info we gor from the forward propagation process.
    :return: the params we updated, weight and bias
    """
    # extract the data we need from what we got in the forward propagation
    x, y = [fprop_cache[key] for key in ('x', 'y')]
    z1, h1, z2, h2 = [fprop_cache[key] for key in ('z1', 'h1', 'z2', 'h2')]
    W1, W2 = [fprop_cache[key] for key in ('W1', 'W2')]

    # calculate the derivatives according to W
    dL_z2 = h2 - y  # dL/dL_z2
    dL_W2 = np.dot(dL_z2, np.transpose(h1))  # dL/dL_z2 * dL_z2/dL_W2

    dh1_z1 = sigmoid(z1) * (1 - sigmoid(z1))  # according to derivative of sigmoid
    dL_z1 = (np.transpose(np.dot(np.transpose(dL_z2), W2))) * dh1_z1  # dL/d_z2 * d_z2/d_h1 * d_h1/d_z1
    # TODO: check the asmatrix on x
    dL_W1 = np.dot(dL_z1, np.transpose(x))  # dL/dL_z2 * dL_z2/dL_h1 * dL_h1/dL_z1 * dL_z1/dL_W1

    # calculate the derivatives according to b
    dL_b2 = dL_z2  # dL/dL_z2 * dL_z2/dL_b2
    dL_b1 = dL_z1  # dL/dL_z2 * dL_z2/dL_h1 * dL_h1/dL_z1 * dL_z1/dL_b1

    return {'b1': dL_b1, 'W1': dL_W1, 'b2': dL_b2, 'W2': dL_W2}


def split_k_fold(k, slice):
    """
    splitting the data to train and validation oin order to be able to check not only on a specific data, but also
    on varied one.
    :param k: how many parts to split it for
    :param slice: the slice for the validation
    :return: the train, its labels, the validation and its labels.
    """
    validation = []
    new_train = []
    new_labels_validation = []
    new_labels = []

    size = len(train_x)
    size_for_part = int(size / k)

    # create the part of validation
    for i in range(0 + size_for_part * slice, 0 + size_for_part * slice + size_for_part):
        validation.append(train_x[i])
        new_labels_validation.append(int(train_y[i]))
    # create the pare of the train
    for j in range(len(train_x)):
        if j in range(0 + size_for_part * slice, 0 + size_for_part * slice + size_for_part):
            continue
        new_train.append(train_x[j])
        new_labels.append(train_y[j])

    return np.asarray(validation), np.asarray(new_train), np.asarray(new_labels), np.asarray(new_labels_validation)


def train(train, y_for_train, W1, b1, W2, b2):
    """
    train the model. run num of epochs and do in this loop forwad propagation and backward propagation in order to
    update the weighs and bias in the best way.
    :param train: the data to kearn
    :param y_for_train: the correct labels
    :param W1: param of weighs of the first layer
    :param b1: param of bias of the first layer
    :param W2: param of weighs of the second layer
    :param b2: param of bias of the second layer
    :return: the params after they updated
    """

    for e in range(EPOCHS):
        # print("In epoc: ", e, "...")
        # shuffle
        my_seed = SEEDS_ARRAY[e]
        np.random.seed(my_seed)
        np.random.shuffle(train_x)
        np.random.seed(my_seed)
        np.random.shuffle(train_y)
        # go over the pictures
        for picture, y in zip(train, y_for_train):
            picture = picture[:, None]
            params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            fprop_cache = fprop(picture, y, params)
            bprop_cache = bprop(fprop_cache)

            d_b1, d_W1, d_b2, d_W2 = [bprop_cache[key] for key in ('b1', 'W1', 'b2', 'W2')]
            # update according to what we got
            W1 = W1 - RATE * d_W1
            W2 = W2 - RATE * d_W2

            b1 = b1 - RATE * d_b1
            b2 = b2 - RATE * d_b2
    # print("trained!")
    return W1, b1, W2, b2


def test(test, y_for_test, W1, b1, W2, b2):
    """
    testing on new data, and printing the accuracy we got, by comparing y_hats we predicted with thw real ones.
    :param test: data to test- validation
    :param y_for_test: labels for validation
    :param W1: param of weighs of the first layer
    :param b1: param of bias of the first layer
    :param W2: param of weighs of the second layer
    :param b2: the params after they updated
    :return: accuracy to calculate in the loop in the main
    """
    # print("start testing...")

    # test
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    acc = 0
    for pic, y_real in zip(test, y_for_test):
        pic = pic[:, None]
        fprop_cache = fprop(pic, y_real, params)
        h2 = fprop_cache['h2']
        # take the label that has the max chance to happen
        y_hat = np.argmax(h2)
        if y_hat == y_real:
            acc += 1

    return acc


def test_real(test, y_for_test, W1, b1, W2, b2):
    """
    testing the data we got, open a file and write to it the y_hats we predicted, each prediction in new line.
    :param test: data to test
    :param y_for_test: nothing (just needed for sending to the fprop)
    :param W1: param of weighs of the first layer
    :param b1: param of bias of the first layer
    :param W2: param of weighs of the second layer
    :param b2: the params after they updated
    :return:nothing
    """
    file = open("test_y", "w")
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    for pic, y_real in zip(test[:-1], y_for_test[:-1]):
        pic = pic[:, None]
        fprop_cache = fprop(pic, y_real, params)
        h2 = fprop_cache['h2']
        y_hat = np.argmax(h2)
        file.write(str(y_hat) + '\n')
    # in order to write the last row without '\n' in the end as we were asked to do
    pic = test[-1]
    pic = pic[:, None]
    fprop_cache = fprop(pic, 0, params)
    h2 = fprop_cache['h2']
    y_hat = np.argmax(h2)
    file.write(str(y_hat))

    file.close()


if __name__ == '__main__':
    training_examples, training_labels, testing_examples = sys.argv[1], sys.argv[2], sys.argv[3]

    train_x = np.loadtxt(training_examples)  # load centroids
    train_y = np.loadtxt(training_labels)  # load centroids
    test_x = np.loadtxt(testing_examples)

    # shuffle the data
    my_seed = FIRST_SEED
    np.random.seed(my_seed)
    np.random.shuffle(train_x)
    np.random.seed(my_seed)
    np.random.shuffle(train_y)

    # normalize the data
    train_x /= 255
    test_x /= 255

    # print("data loaded.")
    # print("start training...")

    # check the hyper parameters according to k-fold

    # accuracy = 0
    # for p in range(PARTS):
    #     W1 = np.random.rand(NUM_NEURONS_IN_LAYER, NUM_PIXELS_OF_PICTURE)
    #     b1 = np.random.rand(NUM_NEURONS_IN_LAYER, 1)
    #     W2 = np.random.rand(NUM_NEURONS_IN_LAST_LAYER, NUM_NEURONS_IN_LAYER)
    #     b2 = np.random.rand(NUM_NEURONS_IN_LAST_LAYER, 1)
    #     # shuffle
    #     my_seed = np.random.randint(1000)
    #     np.random.seed(my_seed)
    #     np.random.shuffle(train_x)
    #     np.random.seed(my_seed)
    #     np.random.shuffle(train_y)
    #     params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    #
    #     validation, new_train, new_labels, new_labels_validation = split_k_fold(K_FOLD, p)
    #     W1, b1, W2, b2 = train(new_train, new_labels, W1, b1, W2, b2)
    #     accuracy += test(validation, new_labels_validation, W1, b1, W2, b2)
    # accuracy /= len(validation) * PARTS
    # print("acc: ", accuracy * 100, "%")

    # for real test
    # initialize with random values
    W1 = np.random.rand(NUM_NEURONS_IN_LAYER, NUM_PIXELS_OF_PICTURE)
    b1 = np.random.rand(NUM_NEURONS_IN_LAYER, 1)
    W2 = np.random.rand(NUM_NEURONS_IN_LAST_LAYER, NUM_NEURONS_IN_LAYER)
    b2 = np.random.rand(NUM_NEURONS_IN_LAST_LAYER, 1)

    # train and test, in the test write to the wanted file.
    W1, b1, W2, b2 = train(train_x, train_y, W1, b1, W2, b2)
    y = np.zeros(len(test_x))
    test_real(test_x, y, W1, b1, W2, b2)
