import numpy as np
import pickle


def unpickle(file):
    """
    Used on loading cifar 10
    :param file: The path where the dataset located
    :return: A dictionary containing the dataset
    """
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def calculate_z(x, w, func_id):
    """
    Calculate the output of the hidden Layer
    :param x: Table N x d which contains our data. N: The number of data that will pass, d: Their dimension
    :param w: Table that contains the weights of the hidden layer
    :param func_id: The id of the function we chose to be as activation
    :return: A table that contains the output of the layer after passing it from the activation
    """
    a = np.dot(x, w.transpose())
    z = activation(func_id=func_id, a=a)

    # Append a column with aces at the start, bias
    z_0 = np.ones((z.shape[0], 1))
    z = np.append(z_0, z, axis=1)

    return z


def softmax(y):
    """
    Compute the softmax function of the output
    :param y: The output of out model
    :return: The probability distribution
    """
    max_of_rows = np.max(y, 1)
    m = np.array([max_of_rows, ] * y.shape[1]).T
    y = y - m
    y = np.exp(y)
    return y / (np.array([np.sum(y, 1), ] * y.shape[1])).T


def feed_forward(w1, w2, x, func_id):
    """
    Performs a feed forward in out network
    :param w1: The weights of the Hidden Layer
    :param w2: The weights of the Output Layer
    :param x: Table N x d which contains our data. N: The number of data that will pass, d: Their dimension
    :param func_id: The id of the function we chose to be as activation
    :return:
    The output of the hidden Layer: z
    The output of the network before passing it through the softmax: softmax_input
    The output of the network after passing it through the softmax: y
    """
    # Hidden layer
    z = calculate_z(x, w1, func_id)

    softmax_input = np.dot(z, np.transpose(w2))

    # Output(soft max) layer returns probabilities
    y = softmax(softmax_input)

    return z, softmax_input, y


def activation(func_id, a):
    """
    Execute the activation on our data
    :param func_id: The id of the function we chose to be as activation
    :param a: Out input that will pass through the activation
    :return: The result of the activation
    """

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
    """
    Execute the derivative of activation on our data
    :param func_id: The id of the function we chose to be as activation
    :param a: Out input that will pass through the derivative of activation
    :return: The result of the derivative of activation
    """

    if func_id == 0:
        # np.exp(a) / np.log(1 + np.exp(a)) causes overflows
        derivative_activation_result = np.exp(np.minimum(0, a)) / (1 + np.exp(- np.abs(a)))

    elif func_id == 1:
        # 1 - (np.exp(a) - np.exp(-a)) ** 2 / (np.exp(a) + np.exp(-a)) ** 2 causes overflows
        derivative_activation_result = 1 - np.power(np.tanh(a), 2)

    else:
        derivative_activation_result = -(np.sin(a))

    return derivative_activation_result


def cost_function(w1, w2, x, t, lambda_val, func_id):
    """
    Compute the cost
    :param w1: The weights of the Hidden Layer
    :param w2: The weights of the Output Layer
    :param x:  Table N x d which contains our data. N: The number of data that will pass, d: Their dimension
    :param t: The real values
    :param lambda_val: The value of lambda (regularization term)
    :param func_id: The id of the function we chose to be as activation
    :return: The cost and the gradients of the wights
    """

    z, softmax_input, y = feed_forward(w1, w2, x, func_id)

    max_error = np.max(softmax_input, axis=1)

    # Compute the cost function to check convergence
    cost = np.sum(t * softmax_input) - np.sum(max_error) - np.sum(np.log(np.sum(
        np.exp(softmax_input - np.array([max_error, ] * softmax_input.shape[1]).T), 1))) - \
        (0.5 * lambda_val) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))

    # calculate gradient of w_2
    w2_grad = (t - y).T.dot(z) - lambda_val * w2

    # Remove the bias
    w2_copy = np.copy(w2[:, 1:])

    # calculate gradient of w_1
    step_1 = derivative_activation(func_id, np.dot(x, np.transpose(w1)))
    step_2 = np.dot(t-y, w2_copy) * step_1
    w1_grad = np.dot(np.transpose(step_2), x) - lambda_val*w1

    return cost, w1_grad, w2_grad


def train(w1, w2, x_train, y_train, lr, train_epochs, train_bs, lambda_val, func_id):
    """
    Perform the training process
    :param w1: The weights of the Hidden Layer
    :param w2: The weights of the Output Layer
    :param x_train:  Table N x d which contains our training data. N: The number of data, d: Their dimension
    :param y_train: Table N x d which contains our training labels
    :param lr: The learning rate
    :param train_epochs: The epochs
    :param train_bs: The batch size
    :param lambda_val: The value of lambda (regularization term)
    :param func_id: The id of the function we chose to be as activation
    :return: The trained weight of the network in order to use them for predictions
    """

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

            # Compute the cost and the grads of the weights
            cost, w1_grad, w2_grad = cost_function(w1=w1, w2=w2,
                                                   x=subset_x, t=subset_y,
                                                   lambda_val=lambda_val, func_id=func_id)

            # Update the weights
            w1 = w1 + lr * w1_grad
            w2 = w2 + lr * w2_grad

    return w1, w2


def predict(w1, w2, x_test, y_test, func_id):
    """
    Performs a complete feed forward on the network in order to make predictions
    :param w1: The weights of the Hidden Layer
    :param w2: The weights of the Output Layer
    :param x_test:  Table N x d which contains our test data. N: The number of data, d: Their dimension
    :param y_test: Table N x d which contains our test labels
    :param func_id: The id of the function we chose to be as activation
    :return: The accuracy of our model in the test data and the number of faults
    """

    # Add the bias on test data
    x0_test = np.ones((x_test.shape[0], 1))
    x_test = np.append(x0_test, x_test, axis=1)

    _, _, pred = feed_forward(w1, w2, x_test, func_id)

    pred = np.argmax(pred, 1)
    real = np.argmax(y_test, 1)

    true_positive = (pred == real).sum()
    accuracy = np.round_(true_positive / x_test.shape[0], decimals=2) * 100
    faults = x_test.shape[0] - true_positive

    print('Faults = {}/{}'.format(faults, real.shape[0]))
    print('Accuracy = {}%'.format(accuracy))

    return accuracy, faults


def grad_check(w1_initial, w2_initial, x, t, lambda_val, func_id):
    """
    Performs a gradients check in order to be sure that our networks works fine
    :param w1_initial: The initial weights of hidden Layer
    :param w2_initial: The initial weights of output Layer
    :param x: Our data
    :param t: The corresponding labels
    :param lambda_val: The value of lambda (regularization term)
    :param func_id: The id of the function we chose to be as activation
    """
    w1 = np.random.rand(*w1_initial.shape)
    w2 = np.random.rand(*w2_initial.shape)
    epsilon = 1e-6

    x0 = np.ones((x.shape[0], 1))
    x = np.append(x0, x, axis=1)

    _list = np.random.randint(x.shape[0], size=5)
    x_sample = np.array(x[_list, :])
    t_sample = np.array(t[_list, :])

    cost, w1_grad, w2_grad = cost_function(w1=w1, w2=w2, x=x_sample, t=t_sample, lambda_val=lambda_val, func_id=func_id)

    print("w1_gradient shape: {} \nw1_shape : {} \nw2_gradient shape: {} \nw2_shape : {} \n".format(
        w1_grad.shape, w1.shape, w2_grad.shape, w2.shape))

    numerical_grad = np.zeros(w1_grad.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numerical_grad
    for k in range(numerical_grad.shape[0]):
        for d in range(numerical_grad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(w1)
            w_tmp[k, d] += epsilon
            e_plus, _, _ = cost_function(w_tmp, w2, x_sample, t_sample, lambda_val, func_id)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(w1)
            w_tmp[k, d] -= epsilon
            e_minus, _, _ = cost_function(w_tmp, w2, x_sample, t_sample, lambda_val, func_id)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numerical_grad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    # Absolute norm
    print("The difference estimate for gradient of w1 is : ", np.max(np.abs(w1_grad - numerical_grad)))

    numerical_grad = np.zeros(w2_grad.shape)
    # Compute all numerical gradient estimates and store them in
    # the matrix numerical_grad
    for k in range(numerical_grad.shape[0]):
        for d in range(numerical_grad.shape[1]):
            # add epsilon to the w[k,d]
            w_tmp = np.copy(w2)
            w_tmp[k, d] += epsilon
            e_plus, _, _ = cost_function(w1, w_tmp, x_sample, t_sample, lambda_val, func_id)

            # subtract epsilon to the w[k,d]
            w_tmp = np.copy(w2)
            w_tmp[k, d] -= epsilon
            e_minus, _, _ = cost_function(w1, w_tmp, x_sample, t_sample, lambda_val, func_id)

            # approximate gradient ( E[ w[k,d] + theta ] - E[ w[k,d] - theta ] ) / 2*e
            numerical_grad[k, d] = (e_plus - e_minus) / (2 * epsilon)

    # Absolute norm
    print("The difference estimate for gradient of w2 is : ", np.max(np.abs(w2_grad - numerical_grad)))


def load_data_set(set_id):
    """
    Loads the data sets
    :param set_id: The id of the dataset you want to load 0 -> MNIST and 1-> CIFAR
    :return:
    """
    print("\nStart Loading, please wait!")
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if set_id == 0:
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

    elif set_id == 1:
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
        train_x, train_y, test_x, test_y = load_data_set(set_id=1)
    else:
        train_x, train_y, test_x, test_y = load_data_set(set_id=0)

    K = 10  # Categories
    M = 100  # Neurons
    epochs = 10  # training epochs
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

    # If you want to perform grad check, uncomment this...
    # grad_check(w1_init, w2_init, train_x, train_y, 0.01, function_id)

    w_1, w_2 = train(w1_init, w2_init, train_x, train_y, learning_rate, epochs, batch_size, 0.1, function_id)

    _, _ = predict(w1=w_1, w2=w_2, x_test=test_x, y_test=test_y, func_id=function_id)
