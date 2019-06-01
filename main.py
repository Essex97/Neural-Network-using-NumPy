import numpy as np
import pickle


# Used for cifar 10
def unpickle(file):
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


# Calculate the output of the hidden Layer
def calculate_z(x, w, func_id):
    a = np.dot(x, w.transpose())
    z = activation(func_id, a)

    # Append a column with aces at the start, bias
    z_0 = np.ones((z.shape[0], 1))
    z = np.append(z_0, z, axis=1)

    return z


# Compute the softmax function of the output
def softmax(y):
    max_of_rows = np.max(y, 1)
    m = np.array([max_of_rows, ] * y.shape[1]).T
    y = y - m
    y = np.exp(y)
    return y / (np.array([np.sum(y, 1), ] * y.shape[1])).T


def feed_forward(w1, w2, x, func_id):
    # Hidden layer
    Z = calculate_z(x, w1, func_id)

    softmax_input = np.dot(Z, np.transpose(w2))

    # Output(soft max) layer returns probabilities
    Y = softmax(softmax_input)

    return Z, softmax_input, Y


def activation(func_id, a):

    if func_id == 0:
        # np.log(1 + np.exp(a)) causes overflows
        activation_result = np.log(1 + np.exp(-np.abs(a))) + np.maximum(a, 0)

    elif func_id == 1:
        # (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a)) causes overflows
        activation_result = np.tanh(a)

    else:
        activation_result = np.cos(a)

    return activation_result


def derivative_activation(func_id, a):

    if func_id == 0:
        # np.exp(a) / np.log(1 + np.exp(a)) causes overflows
        derivative_activation_result = np.exp(np.minimum(0, a)) / (1 + np.exp(- np.abs(a)))

    elif func_id == 1:
        # 1 - (np.exp(a) - np.exp(-a)) ** 2 / (np.exp(a) + np.exp(-a)) ** 2 causes overflows
        derivative_activation_result = 1 - np.power(np.tanh(a), 2)

    else:
        derivative_activation_result = -(np.sin(a))

    return derivative_activation_result


def cost_function(w1, w2, x, t, lamda, func_id):

    Z, softmax_input, Y = feed_forward(w1, w2, x, func_id)

    # Output(soft max) layer returns probabilities
    Y = softmax(softmax_input)

    max_error = np.max(softmax_input, axis=1)

    # Compute the cost function to check convergence
    Ew = np.sum(t * softmax_input) - np.sum(max_error) - \
        np.sum(np.log(np.sum(np.exp(softmax_input - np.array([max_error, ] * softmax_input.shape[1]).T), 1))) - \
        (0.5 * lamda) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    # calculate gradient of w_2
    w2_grad = (t - Y).T.dot(Z) - lamda * w2

    # Remove the bias
    w2_copy = np.copy(w2[:, 1:])

    # calculate gradient of w_1
    step_1 = derivative_activation(func_id, np.dot(x, np.transpose(w1)))
    step_2 = np.dot(t-Y, w2_copy) * step_1
    w1_grad = np.dot(np.transpose(step_2), x) - lamda*w1

    return Ew, w1_grad, w2_grad


def train(w1, w2, x_train, y_train, lr, train_epochs, train_bs, lambda_val, func_id):

    print('Start Training...')

    # Commonly add an ace as the first element: f(x)=1+ax+bx^2...
    x0_train = np.ones((x_train.shape[0], 1))
    x_train = np.append(x0_train, x_train, axis=1)

    # The learning rate needs to be relevant to the batch size
    lr = lr / train_bs

    for i in range(train_epochs):

        # zip the feature and the labels in order to shuffle them in the same order
        zipped = list(zip(x_train, y_train))
        # shuffle them in order to avoid meeting the same examples on the same batches
        np.random.shuffle(zipped)

        x_train, y_train = zip(*zipped)
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # j --> Index of the first element on the batch
        for j in range(0, x_train.shape[0], train_bs):
            subset_x = x_train[j: j+train_bs, :]
            subset_y = y_train[j: j+train_bs, :]

            # Compute the cost
            cost, w1_grad, w2_grad = cost_function(w1=w1, w2=w2,
                                                   x=subset_x, t=subset_y,
                                                   lamda=lambda_val, func_id=func_id)

            # Update the weights
            w1 = w1 + lr * w1_grad
            w2 = w2 + lr * w2_grad

    return w1, w2


def predict(w1, w2, x_test, y_test, func_id):

    # Add the bias on test data
    x0_test = np.ones((x_test.shape[0], 1))
    x_test = np.append(x0_test, x_test, axis=1)

    Z, softmax_input, Y = feed_forward(w1, w2, x_test, func_id)

    # Output(soft max) layer returns probabilities
    pred = softmax(softmax_input)

    pred = np.argmax(pred, 1)
    real = np.argmax(y_test, 1)

    TP = (pred == real).sum()
    accuracy = np.round_(TP/x_test.shape[0], decimals=2) * 100
    FAULTS = x_test.shape[0] - TP

    print('Faults = {}/{}'.format(FAULTS, real.shape[0]))
    print('Accuracy = {}%'.format(accuracy))

    return accuracy, FAULTS


def grad_check(w1_initial, w2_initial, x, t, lambda_val, func_id):
    w1 = np.random.rand(*w1_initial.shape)
    w2 = np.random.rand(*w2_initial.shape)
    epsilon = 1e-6

    x0 = np.ones((x.shape[0], 1))
    x = np.append(x0, x, axis=1)

    _list = np.random.randint(x.shape[0], size=5)
    x_sample = np.array(x[_list, :])
    t_sample = np.array(t[_list, :])

    cost, w1_grad, w2_grad = cost_function(w1=w1, w2=w2, x=x_sample, t=t_sample, lamda=lambda_val, func_id=func_id)

    print("w1_gradient shape: {} \nw1_shape : {} \nw2_gradient shape: {} \nw2_shape : {} \n".format(
        w1_grad.shape, w1.shape, w2_grad.shape, w2.shape))

    numericalGrad = np.zeros(w1_grad.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(w1)
            w_tmp[k, d] += epsilon
            e_plus, _, _ = cost_function(w_tmp, w2, x_sample, t_sample, lambda_val, func_id)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(w1)
            w_tmp[k, d] -= epsilon
            e_minus, _, _ = cost_function(w_tmp, w2, x_sample, t_sample, lambda_val, func_id)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    # Absolute norm
    print("The difference estimate for gradient of w1 is : ", np.max(np.abs(w1_grad - numericalGrad)))

    numericalGrad = np.zeros(w2_grad.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numericalGrad
    for k in range(numericalGrad.shape[0]):
        for d in range(numericalGrad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(w2)
            w_tmp[k, d] += epsilon
            e_plus, _, _ = cost_function(w1, w_tmp, x_sample, t_sample, lambda_val, func_id)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(w2)
            w_tmp[k, d] -= epsilon
            e_minus, _, _ = cost_function(w1, w_tmp, x_sample, t_sample, lambda_val, func_id)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numericalGrad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    # Absolute norm
    print("The difference estimate for gradient of w2 is : ", np.max(np.abs(w2_grad - numericalGrad)))


def load_data_set(mode):
    print("\nStart Loading, please wait!")
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if mode == 0:
        for i in range(10):
            with open('mnistdata/train{0:d}.txt'.format(i), 'r') as train_file:
                for row in train_file:
                    # normalize the pixel values by dividing with 255
                    x_train.append([int(pixel) / 255 for pixel in row.split()])
                    # one hot vector for the category
                    y_train.append([1 if j == i else 0 for j in range(0, 10)])

            with open('mnistdata/test{0:d}.txt'.format(i), 'r') as test_file:
                for row in test_file:
                    # normalize the pixel values by dividing with 255
                    x_test.append([int(pixel) / 255 for pixel in row.split()])
                    # one hot vector for the category
                    y_test.append([1 if j == i else 0 for j in range(0, 10)])

    elif mode == 1:
        for i in range(5):
            train_file = 'cifar-10-batches-py/data_batch_{0:d}'.format(i+1)
            batch_i = unpickle(train_file)
            for j in range(len(batch_i['data'.encode('ascii', 'ignore')])):
                x_train.append(batch_i['data'.encode('ascii', 'ignore')][j])
                label = batch_i['labels'.encode('ascii', 'ignore')][j]
                y_train.append([1 if k == label else 0 for k in range(0, 10)])

        test_file = 'cifar-10-batches-py/test_batch'
        test_batch = unpickle(test_file)
        for j in range(len(test_batch['data'.encode('ascii', 'ignore')])):
            x_test.append(test_batch['data'.encode('ascii', 'ignore')][j])
            label = test_batch['labels'.encode('ascii', 'ignore')][j]
            y_test.append([1 if k == label else 0 for k in range(0, 10)])

    print('They were loaded\n')
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


if __name__ == '__main__':

    function_id = input("\nEnter the id of the activation function (default=cos(a)): \n"
                        "id | function \n"
                        "--------------\n"
                        "0  | log(1+exp(a) \n"
                        "1  | tanh \n"
                        "2  | cos(a) \n"
                        "Your choice: ")

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

    if data_set_id == '1':
        train_x, train_y, test_x, test_y = load_data_set(mode=1)
    else:
        train_x, train_y, test_x, test_y = load_data_set(mode=0)

    K = 10  # Categories
    M = 200  # Neurons
    epochs = 20  # training epochs
    batch_size = 100  # batch size
    D = train_x.shape[1]  # Number of features

    print('Train x: {} Test x: {}'.format(train_x.shape, test_x.shape))
    print('Train y: {} Test y: {}'.format(train_y.shape, test_y.shape))

    # Initialize the weights of the model
    center = 0
    s = 1 / np.sqrt(D + 1)
    w1_init = np.random.normal(center, s, (M, D + 1))
    w2_init = np.zeros((K, M + 1))

    # Add an ace for the zero degree element of the logistic regression function : f(x)=1+ax+bx^2...
    w1_init[:, 0] = 1
    w2_init[:, 0] = 1

    # grad_check(w1_init, w2_init, train_x, train_y, 0.01, function_id)

    w_1, w_2 = train(w1_init, w2_init, train_x, train_y, learning_rate, epochs, batch_size, 0.1, function_id)

    _, _ = predict(w1=w_1, w2=w_2, x_test=test_x, y_test=test_y, func_id=function_id)
