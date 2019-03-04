import numpy as np
from math import exp, cos


def cost_function(w, l, y, t):
    E = 0
    for n in range(y.shape[0]):
        for k in range(y.shape[1]):
            E += t[n, k] * np.log(y[n, k])

    return E - l / 2 * (np.linalg.norm(w, 2))


def get_y(w, k, z):
    numerator = exp(np.dot(np.transpose(w[k]), z))
    denominator = np.sum(exp(np.dot(np.transpose(w), z)), axis=1)
    return numerator / denominator


def load_data_set(mode):
    print("\nStart Loading, please wait!")
    train_set = []
    test_set = []

    if mode == 0:
        for i in range(10):
            with open('mnistdata/train{0:d}.txt'.format(i), 'r') as train_file:
                for row in train_file:
                    train_set.append([int(pixel) / 255 for pixel in row.split()])

            with open('mnistdata/test{0:d}.txt'.format(i), 'r') as test_file:
                for row in test_file:
                    test_set.append([int(pixel) / 255 for pixel in row.split()])

    return np.array(train_set), np.array(test_set)


if __name__ == '__main__':
    function_id = input("\nEnter the id of the activation function (default=cos(a)): \n"
                        "id | function \n"
                        "--------------\n"
                        "0  | log(1+exp(a) \n"
                        "1  | exp(a) - exp(-a) / exp(a) + exp(-a) \n"
                        "2  | cos(a) \n"
                        "Your choice: ")

    if function_id == 0:
        activation = lambda a: np.log(1 + exp(a))
    elif function_id == 1:
        activation = lambda a: (exp(a) - exp(-a)) / (exp(a) + exp(-a))
    else:
        activation = lambda a: cos(a)

    learning_rate = input("\nEnter the learning rate (default=0.01): ")
    try:
        learning_rate = float(learning_rate)
    except ValueError:
        learning_rate = 0.01

    data_set_id = input("\nEnter the id of the data set (default=MNIST): \n"
                        "id | dataset \n"
                        "-------------\n"
                        "0  | MNIST \n"
                        "1  | CIFAR-10 \n"
                        "Your choice: ")

    if data_set_id == 1:
        train, test = load_data_set(mode=1)
    else:
        train, test = load_data_set(mode=0)

    y = np.random.rand(5, 3)
    t = np.random.randint(2, size=(5, 3))
    W = np.random.rand(3, 2)
    print(cost_function(W, learning_rate, y, t))
