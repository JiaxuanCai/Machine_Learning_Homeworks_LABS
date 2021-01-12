import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Logistic


class WatermelonLogistic:
    data = []
    dataX = []
    dataY = []
    data_train_X = []
    data_train_Y = []
    data_test_X = []
    data_test_Y = []

    def __init__(self):
        self.data_get()
        self.data_split()
        self.model_train(0.05, 0.00000001, 100000)

    def data_get(self):
        self.data = np.array(pd.read_csv('data/watermelon_3a.csv'))
        self.dataX = self.data[:, np.arange(1, 3)]
        self.dataY = np.transpose([self.data[:, 3]])

    def data_split(self):
        # 由于数据集很小，所以使用自助法，通过随机抽样的方式划分训练集与测试集
        # 通过产生的随机数获得抽取样本的序号
        bootstrapping = []
        for i in range(len(self.data)):
            bootstrapping.append(np.floor(np.random.random() * len(self.data)))
        # 通过序号获得原始数据集中的数据
        D_1 = []  # 训练集
        for i in range(len(self.data)):
            D_1.append(self.data[int(bootstrapping[i])])

        l = []  # 用l存储a中b的每一行的索引位置
        for i in range(len(np.array(D_1))):
            for j in range(len(self.data)):
                if list(self.data[j]) == list(np.array(D_1)[i]):  # op.eq比较两个list，相同返回Ture
                    l.append(j)
        # delete函数删除数据集中对应行
        D_2 = np.delete(self.data, l, axis=0)
        self.data_train_X = np.array(D_1)[:, np.arange(1, 3)]
        self.data_train_Y = np.transpose([np.array(D_1)[:, 3]])
        self.data_test_X = np.array(D_2)[:, np.arange(1, 3)]
        self.data_test_Y = np.transpose([np.array(D_2)[:, 3]])

    def model_train(self, alpha, epsilon, maxloop):
        # 训练模型
        m = len(self.data_train_X)
        X = np.concatenate((np.ones((m, 1)), self.data_train_X), axis=1)
        Y = self.data_train_Y
        testX = np.concatenate((np.ones((len(self.data_test_X), 1)), self.data_test_X), axis=1)
        testY = self.data_test_Y
        thetas, costs, iterationCount, accuracy = Logistic.gradient(X, Y, testX, testY, alpha, epsilon, maxloop)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 绘制分类结果
        for i in range(len(self.data_train_X)):
            x = np.concatenate((np.ones((len(self.data_train_X), 1)), self.data_train_X), axis=1)[i]
            if self.data_train_Y[i] == 1:
                plt.scatter(x[1], x[2], marker='*', color='blue', s=50)
            else:
                plt.scatter(x[1], x[2], marker='o', color='green', s=50)

        for i in range(len(self.data_test_X)):
            x = np.concatenate((np.ones((len(self.data_test_X), 1)), self.data_test_X), axis=1)[i]
            if self.data_test_Y[i] == 1:
                plt.scatter(x[1], x[2], marker='*', color='red', s=50)
            else:
                plt.scatter(x[1], x[2], marker='o', color='orange', s=50)

        hSpots = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
        theta0, theta1, theta2 = thetas[-1]

        vSpots = -(theta0 + theta1 * hSpots) / theta2
        plt.plot(hSpots, vSpots, color='red', linewidth=.5)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('分类结果图示')
        plt.show()

        # 绘制代价随着迭代次数的变化情况
        plt.plot(range(len(costs)), costs)
        plt.xlabel(u'迭代次数')
        plt.ylabel(u'代价J')
        plt.title("代价随迭代次数的变化")

        # 绘制各预测参数theta随迭代次数变化
        thetasFig, ax = plt.subplots(len(thetas[0]))
        thetas = np.asarray(thetas)
        for idx, sp in enumerate(ax):
            thetaList = thetas[:, idx]
            sp.plot(range(len(thetaList)), thetaList)
            sp.set_xlabel('Number of iteration')
            sp.set_ylabel(r'$\theta_%d$' % idx)
        plt.show()

        # 绘制分类准确率随迭代次数的变化
        plt.plot(range(len(accuracy)), accuracy)
        plt.title('分类准确率随迭代次数的变化')
        plt.xlabel(u'迭代次数')
        plt.ylabel(u'测试集上的预测准确率')
        print('准确率为')
        print(accuracy[len(accuracy)-1])
        print('模型参数为')
        print(thetas[len(thetas) - 1])



if __name__ == '__main__':
    test = WatermelonLogistic()
    plt.show()
