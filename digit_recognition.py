import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def relu(z):
    a = np.ndarray((len(z), 1))
    for i in range(len(z)):
        a[i] = max(0, z[i])
    return a

def relu_deriv(vec):
    a = np.ndarray((len(vec), 1))
    for i in range(len(vec)):
        if vec[i] > 0:
            a[i] = 1
        else:
            a[i] = 0
    return a


def softmax(z):
    s = np.exp(z)
    return s / s.sum()

def softmax_deriv(s):
    n = len(s)
    jacobi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                jacobi[i][j] = -(s[i] * s[j])
            else:
                jacobi[i][j] = (s[i] * (1 - s[j]))
    return jacobi


def xent_cost(s, y):
    cost = 0
    for i in range(len(s)):
        cost = cost + np.nan_to_num(-y[i] * np.log(s[i]))
    return cost[0]

def xent_deriv(s, y):
    return - y / s

def xent_softmax_deriv(s, y):
    return np.dot(softmax_deriv(s), xent_deriv(s, y))

def xent_softmax_deriv2(s, y):
    return s - y 


def vectorized_y(j, size):
    e = np.zeros((size, 1))
    e[j] = 1.0
    return e

def vectorized_x(j):
    e = np.ndarray((len(j), 1))
    for i in range(len(j)):
        e[i] = j[i]
    return e


def read_data(path, size):
    train = pd.read_csv(path)
    x_cols = train.columns[1:]
    y_cols = train.columns[0]
    X = train[x_cols].to_numpy()
    X = X / 255 * 2 - 1
    y = train[y_cols].to_numpy()
    data = [(vectorized_x(x), vectorized_y(y, size)) for x, y in zip(X, y)]
    return data 


def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


class Network(object):


    def __init__(self, size):
        self.num_layers = len(size)
        self.size = size
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(size[:-1], size[1:])]
        self.biases = [np.random.randn(y, 1) for y in size[1:]]


    def __str__(self):
        desc ='--------------------------\n'
        for i in range(len(self.size)):
            desc = desc + f'Layer {i + 1}:\t {self.size[i]} units\n'
        x = 0
        for j in range(len(self.size) - 1):
            x = x + self.size[j] * self.size[j + 1] + self.size[j + 1]
        desc = desc + f'--------------------------\nTotal params: {x}'
        return desc


    def predict(self, a):
        for i in range(self.num_layers - 2):
            a = relu(np.dot(self.weights[i], a) + self.biases[i])
        a = softmax(np.dot(self.weights[-1], a) + self.biases[-1])
        return a


    def fit(self, train_data, epochs, mini_batch_size, test_data=None, lr = 0.001, decay = 0.8):
        n = len(train_data)
        lrate = []
        loss = []
        val_loss = []
        accuracy = []
        val_accuracy = []
        print ("Random accuracy = {0} ".format(self.accuracy(test_data)))
        for j in range(epochs):
            if j > 2 and val_accuracy[-1] < val_accuracy[-2] and val_accuracy[-1] < val_accuracy[-3]:
                lr = lr * 0.5
                print('Reducing learning rate to {0}'.format(lr))
            random.shuffle(train_data)
            mini_batches = [
                train_data[k:k+mini_batch_size]
                for k in range(0, n - (n % mini_batch_size), mini_batch_size)]
            for mini_batch in mini_batches:
                self.nesterov(mini_batch, lr, decay)
            if test_data:
                val_accuracy.append(self.accuracy(test_data))
                val_loss.append(self.loss(test_data))
                print ("Epoch {0}: val accuracy = {1} ".format(
                    j + 1, val_accuracy[-1]))
            else:
                print ("Epoch {0} complete".format(j))
            lrate.append(lr)
            accuracy.append(self.accuracy(train_data))
            loss.append(self.loss(train_data))
        return np.array([lrate, loss, val_loss, accuracy, val_accuracy], dtype=object)


    def nesterov(self, mini_batch, lr, decay):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        n = self.num_layers - 1
        for x, y in mini_batch:
            prev_weights = self.weights
            prev_biases = self.biases
            for i in range(n):
                self.weights[i] = self.weights[i] - decay * grad_w[i]
                self.biases[i] = self.biases[i] - decay * grad_b[i]
            delta_w, delta_b = self.gradient(x, y)
            self.weights = prev_weights
            self.biases = prev_biases
            for i in range(n):
                grad_w[i] = decay * grad_w[i] + lr * delta_w[i]
                grad_b[i] = decay * grad_b[i] + lr * delta_b[i]
                self.weights[i] = self.weights[i] - grad_w[i]
                self.biases[i] = self.biases[i] - grad_b[i]
                       

    def momentum(self, mini_batch, lr, decay):
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        n = self.num_layers - 1
        for x, y in mini_batch:
            delta_w, delta_b = self.gradient(x, y)
            for i in range(n):
                grad_w[i] = decay * grad_w[i] + lr * delta_w[i]
                grad_b[i] = decay * grad_b[i] + lr * delta_b[i]
                self.weights[i] = self.weights[i] - grad_w[i]
                self.biases[i] = self.biases[i] - grad_b[i]

    
    def gradient(self, x, y):
        #foward
        a = x
        zi = []
        ai = [a]
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, a)+b
            a = relu(z)
            zi.append(z)
            ai.append(a)

        z = np.dot(self.weights[-1], a)+self.biases[-1]
        a = softmax(z)
        zi.append(z)
        ai.append(a)
        #backward
        deriv_z = [np.zeros(b.shape) for b in self.biases]
        deriv_b = [np.zeros(b.shape) for b in self.biases]
        deriv_w = [np.zeros(w.shape) for w in self.weights]

        deriv_z[-1] = xent_softmax_deriv2(ai[-1], y)
        deriv_b[-1] = deriv_z[-1]
        deriv_w[-1] = np.dot(deriv_z[-1], ai[-2].transpose())
        for l in range(2, self.num_layers):
            deriv_z[-l] = np.dot(self.weights[-l+1].transpose(), deriv_z[-l+1]) * relu_deriv(zi[-l])
            deriv_b[-l] = deriv_z[-l]
            deriv_w[-l] = np.dot(deriv_z[-l], ai[-l-1].transpose())
        return (deriv_w, deriv_b)


    def accuracy(self, data):
        n = len(data)
        results = [(np.argmax(self.predict(x)), np.argmax(y))
                        for x, y in data]
        acc = sum(int(x == y) for x, y in results) / n
        return acc

    def loss(self, data):
        n = len(data)
        results = [(self.predict(x), y)
                        for x, y in data]
        loss = sum(xent_cost(x, y) for x, y in results) / n
        return loss

    def save(self, filename):
        data = {"sizes": self.size,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

        
if '__main__' == __name__:

    size = [784, 128, 11]

    train_path = "./mnist_train.csv"
    train_data = read_data(train_path, size[-1])
    test_path = "./mnist_test.csv"
    test_data = read_data(test_path, size[-1])
    print('Data loaded')


    net = Network(size)
    history = net.fit(train_data, 10, 64, test_data, 0.0015)
    path = './network.json'
    net.save(path)


    history_frame = pd.DataFrame(history).T
    history_frame.columns = ['lrate', 'loss', 'val_loss', 'accuracy', 'val_accuracy']
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    history_frame.loc[:, ['loss', 'val_loss']].plot(ax = axes[0])
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot(ax = axes[1])
    axes[0].set_title("Loss plot")
    axes[1].set_title("Accuracy plot")
    plt.show()
