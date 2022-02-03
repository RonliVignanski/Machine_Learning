
import numpy as np
import sys

PA_SEED = 5357
PER_SEED = 859400
SVM_SEED = 3481

def canberra_distance(first_point, second_point):
    """
    calculate the distance from two points
    :param first_point: the new point we want to predict her label
    :param second_point: the point from the trained data
    :return: the distance
    """
    sum = 0
    for i, j in zip(first_point, second_point):

        numerator = np.abs(i - j)
        denominator = np.abs(i) + np.abs(j)
        if denominator == 0:
            continue
        sum = sum + numerator / denominator

    return sum


def put_data_in_dict(dictionary, data):
    """
    insert the data to a dictionary in order to have an easy way to go the columns of the file
    :param dictionary: dictionary to save the data
    :param data: the data nedede to be saved
    :return: nothing
    """
    for tr in data:
        for index in range(len(tr)):
            dictionary[index] = []
        break

    for tr in data:
        for index in range(len(tr)):
            dictionary[index].append(tr[index])


def normalization(dictionary, data, type):
    """
    normalize the data
    :param dictionary: to norm according to the data that in this dictionary
    :param data: what needed to be normalized
    :param type: min_max or z_score
    :return: nothing
    """
    list_info = []
    for i in dictionary.keys():
        minimum = np.min(dictionary[i])
        maximum = np.max(dictionary[i])
        mean = np.mean(dictionary[i])
        stand_dev = np.std(dictionary[i])
        list_info.append((minimum, maximum, mean, stand_dev))

    if type == "min_max":
        for row in range(len(data)):
            for col in range(len(data[row])):
                diff_1 = data[row][col] - list_info[col][0]
                diff_2 = list_info[col][1] - list_info[col][0]
                if diff_2 == 0:
                    data[row][col] = 0
                    continue
                result = (diff_1 / diff_2) * (1 - 0) + 0
                data[row][col] = result
    elif type == "z_score":
        for row in range(len(data)):
            for col in range(len(data[row])):
                diff_1 = data[row][col] - list_info[col][2]
                diff_2 = list_info[col][3]
                if diff_2 == 0:
                    data[row][col] = 0
                    continue
                result = diff_1 / diff_2
                data[row][col] = result
    else:
        raise ValueError('the option of normalization invalid')


def knn(test, train, labels, k, distance):
    """
    find the labels of the train according to the model we trained
    :param test: what to test
    :param train: according to what we need to learn
    :param labels: correct labels of the train
    :param k: number of neighbors that we put a label according to them
    :param distance: type of metric
    :return: labels of the trained model
    """
    labels_p = []

    for p in test:
        distances = []
        num_0 = 0
        num_1 = 0
        num_2 = 0
        for row in range(len(train)):
            d = distance(p, train[row])
            distances.append((d, row))
        sorted_d = sorted(distances)

        for i in range(k):
            label = labels[sorted_d[i][1]]
            if label == 0:
                num_0 = num_0 + 1
            if label == 1:
                num_1 = num_1 + 1
            if label == 2:
                num_2 = num_2 + 1

        # check which label showed the most of the time
        if num_0 == max(num_0, num_1, num_2):
            labels_p.append(0)
        elif num_1 == max(num_0, num_1, num_2):
            labels_p.append(1)
        elif num_2 == max(num_0, num_1, num_2):
            labels_p.append(2)

    return labels_p


def perceptron(num_of_epochs, labels, train):
    """
    train a data set according to the updating rules of perceptron, with given hyper parameters
    :param num_of_epochs: max number of epochs
    :param labels: correct labels of the train
    :param train: the data we learn
    :return: weighs
    """
    # add bias
    train = np.hstack((np.ones((train.shape[0], 1)), train))

    num_classes = 3
    num_features = len(train[0])
    weights = np.zeros((num_classes, num_features))
    rate = 0.1
    for i in range(num_of_epochs):
        # seed
        np.random.seed(PER_SEED)
        # shuffle the samples
        train_x_y = list(zip(train, labels))
        np.random.shuffle(train_x_y)
        train, labels = zip(*train_x_y)

        for r in range(len(train)):
            y_hat = np.argmax(np.dot(weights, train[r]))
            if y_hat != labels[r]:
                weights[int(labels[r])] = weights[int(labels[r])] + rate * train[r]
                weights[int(y_hat)] = weights[int(y_hat)] - rate * train[r]

    return weights


def predict(weights, data):
    """
    find the labels according to the learning we have done
    :param weights: the distances between the points to the seperated lines of the classes
    :param data: the test we want to give them labels
    :return: the labels we gave to the test points
    """
    y_hats = []
    for r in range(len(data)):
        y_hat = np.argmax(np.dot(weights, data[r]))
        y_hats.append(y_hat)
    return y_hats


def hinge_loss(w_y_hat, w_h, train_data):
    """
    to calculate for tau in passive agressive
    :param w_y_hat: weighs of the predicted labels
    :param w_h: weighs of the real labels
    :param train_data: data to learn on
    :return: hinge loss
    """
    return max(0, 1 - np.dot(w_h, train_data) + np.dot(w_y_hat, train_data))


def pa(num_of_epochs, labels, train):
    """
    train a data set according to the updating rules of passive aggressive multi class, with given hyper parameters
    :param num_of_epochs: max epochs
    :param labels: correct labels of the train
    :param train: the data we learn
    :return: weighs
    """
    # add bias
    train = np.hstack((np.ones((train.shape[0], 1)), train))

    num_classes = 3
    num_features = len(train[0])
    weights = np.zeros((num_classes, num_features))
    tao = 0
    for i in range(num_of_epochs):
        # seed
        np.random.seed(PA_SEED)
        # shuffle the samples
        train_x_y = list(zip(train, labels))
        np.random.shuffle(train_x_y)
        train, labels = zip(*train_x_y)
        for row in range(len(train)):

            distances = np.dot(weights, train[row])
            distances = np.delete(distances, int(labels[row]))
            r = np.argmax(distances)
            if r >= labels[row]:
                r += 1
            loss_for_update = max(0,
                                  1 - np.dot(weights[int(labels[row])], train[row]) + np.dot(weights[int(r)],
                                                                                             train[row]))
            # update when there is a mistake or it is a good label but not the correct margin
            if loss_for_update > 0:
                tao = hinge_loss(weights[int(r)], weights[int(labels[row])], train[row]) / (2 * (
                        np.linalg.norm(train[row]) ** 2))
                weights[int(labels[row])] = weights[int(labels[row])] + tao * train[row]
                weights[int(r)] = weights[int(r)] - tao * train[row]
    return weights


def svm(num_of_epochs, labels, train):
    """
    train a data set according to the updating rules of svm multi class, with given hyper parameters
    :param num_of_epochs: max epochs
    :param labels: correct labels of the train
    :param train: the data we learn
    :return: weighs
    """
    # bias
    train = np.hstack((np.ones((train.shape[0], 1)), train))

    num_classes = 3
    num_features = len(train[0])
    weights = np.zeros((num_classes, num_features))
    rate = 0.1
    lamda = 0.01
    for i in range(num_of_epochs):
        # seed
        np.random.seed(SVM_SEED)
        # shuffle the samples
        train_x_y = list(zip(train, labels))
        np.random.shuffle(train_x_y)
        train, labels = zip(*train_x_y)
        for row in range(len(train)):
            distances = np.dot(weights, train[row])
            distances = np.delete(distances, int(labels[row]))
            r = np.argmax(distances)
            if r >= labels[row]:
                r += 1
            loss_for_update = max(0,
                                  1 - np.dot(weights[int(labels[row])], train[row]) + np.dot(weights[int(r)],
                                                                                             train[row]))
            # update any time
            weights[0] = (1 - rate * lamda) * weights[0]
            weights[1] = (1 - rate * lamda) * weights[1]
            weights[2] = (1 - rate * lamda) * weights[2]

            # update when there is a mistake or it is a good label but not the correct margin
            if loss_for_update > 0:
                weights[int(labels[row])] = weights[int(labels[row])] + rate * train[row]
                weights[int(r)] = weights[int(r)] - rate * train[row]

    return weights


def delete_col(data, c):
    """
    delete col from the data
    :param data: data to delete from
    :param c: the num of col
    :return: data without the deleted colomn
    """
    return np.delete(data, c, axis=1)


if __name__ == '__main__':
    training_examples, training_labels, testing_examples, output = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    train_x = np.loadtxt(training_examples, delimiter=',')  # load centroids
    train_y = np.loadtxt(training_labels, delimiter=',')  # load centroids
    test_x = np.loadtxt(testing_examples, delimiter=',')  # load centroids

    # normalize the data
    dictionary_test = dict()
    # putting the test data in a dictionary and normalize it
    put_data_in_dict(dictionary_test, train_x)
    normalization(dictionary_test, test_x, "z_score")

    dictionary = dict()
    # putting the train data in a dictionary and normalize it
    put_data_in_dict(dictionary, train_x)
    normalization(dictionary, train_x, "z_score")

    train_x = delete_col(train_x, 4)
    test_x = delete_col(test_x, 4)

    f = open(output, "w")
    labels = knn(test_x, train_x, train_y, 9, canberra_distance)

    w_per = perceptron(15, train_y, train_x)

    w_pa = pa(16, train_y, train_x)

    w_svm = svm(17, train_y, train_x)

    # add bias to the test
    test_x = np.hstack((np.ones((test_x.shape[0], 1)), test_x))

    y_hats_per = predict(w_per, test_x)
    y_hats_pa = predict(w_pa, test_x)
    y_hats_svm = predict(w_svm, test_x)

    # writing to the file
    for i in range(len(y_hats_per)):
        f.write(f"knn: {labels[i]}, perceptron: {y_hats_per[i]}, svm: {y_hats_svm[i]}, pa: {y_hats_pa[i]}\n")
    f.close()
