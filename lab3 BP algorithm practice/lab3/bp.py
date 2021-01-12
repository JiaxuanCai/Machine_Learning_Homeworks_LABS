import numpy as np
import pandas as pd
import tools
import matplotlib.pyplot as plt

def train_zoo():
    data = np.array(pd.read_csv('data/zoo.csv'))
    x_ori = data[:, np.arange(1, 17)]
    x_ori = x_ori.astype(np.float)
    y_ori = data[:, np.arange(17, 18)]
    y_ori = y_ori.astype(np.uint8)
    X = np.mat(x_ori)
    y = np.mat(y_ori)

    accus = []

    for i in range(0,500):
        weights = tools.train(X, y, num_layer=1, num_nodes=25, regularization_index=0.01, iters=i)
        def readable_predict(X, thetas):
            lens = len(np.array(y))
            count = 0
            for idx in range(0, lens - 1):
                print('predict:', (np.argmax(tools.compute_output(thetas,X[idx])[-1]) + 1))
                print('real tag:', np.array(y)[idx][0])
                if ((np.argmax(tools.compute_output(thetas,X[idx])[-1]) + 1) == np.array(y)[idx][0]):
                    count += 1
            acc = count / lens
            accus.append(acc)
            print("准确率为", acc)
        readable_predict(X, weights)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制准确率随着迭代次数的变化情况
    plt.plot(range(len(accus)), accus)
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'准确率')
    plt.title("准确率随迭代次数的变化")
    plt.show()


def train_iris():
    data = np.array(pd.read_csv('data/iris.csv'))
    X = data[:, np.arange(1, 5)]
    X = X.astype(np.float)
    X = np.mat(X)
    y = data[:, np.arange(5, 6)]
    y[y == 'Iris-setosa'] = np.uint8(1)
    y[y == 'Iris-versicolor'] = np.uint8(2)
    y[y == 'Iris-virginica'] = np.uint8(3)
    y = y.astype(np.uint8)
    y = np.mat(y)

    accus = []

    for i in range(0,500):
        weights = tools.train(X, y, num_layer=1, num_nodes=25, regularization_index=0.01, iters=i)
        def readable_predict(X, thetas):
            lens = len(np.array(y))
            count = 0
            for idx in range(0, lens - 1):
                print('predict:', (np.argmax(tools.compute_output(thetas,X[idx])[-1]) + 1))
                print('real tag:', np.array(y)[idx][0])
                if ((np.argmax(tools.compute_output(thetas,X[idx])[-1]) + 1) == np.array(y)[idx][0]):
                    count += 1
            acc = count / lens
            accus.append(acc)
            print("准确率为", acc)
        readable_predict(X, weights)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制准确率随着迭代次数的变化情况
    plt.plot(range(len(accus)), accus)
    plt.xlabel(u'迭代次数')
    plt.ylabel(u'准确率')
    plt.title("准确率随迭代次数的变化")
    plt.show()

if __name__ == '__main__':
    train_zoo()
    # train_iris()
