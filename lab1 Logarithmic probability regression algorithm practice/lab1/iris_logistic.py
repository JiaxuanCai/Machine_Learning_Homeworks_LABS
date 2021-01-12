import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Logistic


class IrisLogistic:
    test_X = []
    test_Y = []

    train_X = []
    train_Y1 = []

    def __init__(self):
        self.data_get()
        # self.loop_train()
        self.vote(2000)


    def data_get(self):
        data = np.array(pd.read_csv('data/iris.csv'))

        data1 = data[np.where(data[:, 5] == 'Iris-setosa')]
        data2 = data[np.where(data[:, 5] == 'Iris-versicolor')]
        data3 = data[np.where(data[:, 5] == 'Iris-virginica')]

        r1 = np.random.permutation(data1)
        test1 = r1[np.arange(0, 10), :]
        test1_X = test1[:, np.arange(1, 5)]
        test1_Y = np.transpose([test1[:, 5]])
        train1 = r1[np.arange(10, 50), :]
        train1_X = train1[:, np.arange(1, 5)]
        train1_Y = np.transpose([train1[:, 5]])

        r2 = np.random.permutation(data2)
        test2 = r2[np.arange(0, 10), :]
        test2_X = test2[:, np.arange(1, 5)]
        test2_Y = np.transpose([test2[:, 5]])
        train2 = r2[np.arange(10, 50), :]
        train2_X = train2[:, np.arange(1, 5)]
        train2_Y = np.transpose([train2[:, 5]])

        r3 = np.random.permutation(data3)
        test3 = r3[np.arange(0, 10), :]
        test3_X = test3[:, np.arange(1, 5)]
        test3_Y = np.transpose([test3[:, 5]])
        train3 = r3[np.arange(10, 50), :]
        train3_X = train3[:, np.arange(1, 5)]
        train3_Y = np.transpose([train3[:, 5]])

        self.train_X = np.concatenate((train1_X, train2_X, train3_X), axis=0)
        self.train_Y = np.concatenate((train1_Y, train2_Y, train3_Y), axis=0)
        self.test_X = np.concatenate((test1_X, test2_X, test3_X), axis=0)
        self.test_Y = np.concatenate((test1_Y, test2_Y, test3_Y), axis=0)

    def model_train(self, alpha, epsilon, maxloop):
        # 训练三个模型
        m1 = len(self.train_X)
        X = np.concatenate((np.ones((m1, 1)), self.train_X), axis=1).astype(np.float64)
        Y = self.train_Y
        Y_temp1 = Y.copy()
        Y_temp2 = Y.copy()
        Y_temp3 = Y.copy()


        Y_temp1[self.train_Y != 'Iris-setosa'] = np.float64(0)
        Y_temp1[self.train_Y == 'Iris-setosa'] = np.float64(1)
        Y1 = Y_temp1.astype(np.float64)
        thetas1, costs1, iterationCount1 = Logistic.gradient2(X, Y1,  alpha, epsilon, maxloop)


        Y_temp2[self.train_Y == 'Iris-setosa'] = np.float64(0)
        Y_temp2[self.train_Y == 'Iris-virginica'] = np.float64(0)
        Y_temp2[self.train_Y == 'Iris-versicolor'] = np.float64(1)
        Y2 = Y_temp2.astype(np.float64)
        thetas2, costs2, iterationCount2 = Logistic.gradient2(X, Y2, alpha, epsilon, maxloop)


        Y_temp3[self.train_Y == 'Iris-setosa'] = np.float64(0)
        Y_temp3[self.train_Y == 'Iris-versicolor'] = np.float64(0)
        Y_temp3[self.train_Y == 'Iris-virginica'] = np.float64(1)
        Y3 = Y_temp3.astype(np.float64)
        thetas3, costs3, iterationCount3 = Logistic.gradient2(X, Y3, alpha, epsilon, maxloop)

        return thetas1, thetas2, thetas3

    def loop_train(self):
        accuracies = []
        for i in range(1, 100):
            accu = self.vote(int(i))
            accuracies.append(accu)

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(accuracies)
        plt.title('Iris数据集分类准确率随迭代次数的变化情况')
        plt.xlabel('迭代次数')
        plt.ylabel('分类准确率')
        plt.show()



    def vote(self, looptime):
        thetas1, thetas2, thetas3 = self.model_train(0.05, 0.00000001, looptime)
        X = self.test_X.copy()
        X = np.concatenate((np.ones((30, 1)), X), axis=1).astype(np.float64)
        vote1 = Logistic.sigmoid(np.dot(X, thetas1[len(thetas1)-1]))
        vote2 = Logistic.sigmoid(np.dot(X, thetas2[len(thetas2)-1]))
        vote3 = Logistic.sigmoid(np.dot(X, thetas3[len(thetas3)-1]))
        result = []
        for i in range(0, 30):
            max_vote = max(vote1[i], vote2[i], vote3[i])
            if max_vote == vote1[i]:
                result.append(1)
            elif max_vote == vote2[i]:
                result.append(2)
            else:
                result.append(3)
        realistic = []
        for j in range(0, 30):
            if self.test_Y[j] == 'Iris-setosa':
                realistic.append(1)
            elif self.test_Y[j] == 'Iris-versicolor':
                realistic.append(2)
            else:
                realistic.append(3)

        hit = 0
        for k in range(0, 30):
            if result[k] == realistic[k]:
                hit += 1

        accuracy = 1.0 * hit / 30
        print("测试集上的预测结果为")
        print(result)

        print("真实的分类为")
        print(realistic)

        print("准确率为")
        print(accuracy)
        return accuracy


if __name__ == '__main__':
    test = IrisLogistic()


