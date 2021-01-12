import numpy as np


# sigmoid函数的定义
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# 损失函数J的定义
def J(theta, array_x, array_y, my_lambda=0):
    m, n = array_x.shape  # X的尺寸
    h = sigmoid(np.dot(array_x, theta))  # 调用sigmoid函数求h
    J = (-1.0 / m) * (np.dot(np.log(h).T, array_y) + np.dot(np.log(1 - h).T, 1 - array_y)) + (my_lambda / (2.0 * m)) * np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return np.inf  # J为空的特殊情况返回无穷大
    return J.flatten()[0]  # 否则返回第一项的值即为求出的损失函数值


# 预测根据X对其标记进行预测
def predict(array_x, theta):
    predictResult = sigmoid(np.dot(array_x, theta))
    predictResult[predictResult >= 0.5] = 1  # 根据sigmoid函数定义与性质，大于0.5即预测为1
    predictResult[predictResult < 0.5] = 0  # 否则预测为0
    return predictResult


# 对标记值与预测值进行比较，求出预测准确率
def accuracy_rate(array_y, pre_y):
    n = len(pre_y)
    accNum = 0.0
    for i in range(n):
        if pre_y[i] == array_y[i]:
            accNum = accNum + 1
    return accNum / float(n)


# 梯度下降法优化损失函数的核心算法
def gradient(array_x, array_y, test_X, test_Y, alpha, epsilon, max_loop):
    m = len(array_x)
    n = 3
    # 矩阵尺寸的定义
    theta = np.zeros((n, 1))
    cost = J(theta, array_x, array_y)  # 求当前损失函数的值
    costs = [cost]
    thetas = [theta]
    accuracy = 0.0
    accuracies = [accuracy]
    my_Lambda = float(0)  # 需将拉姆达定义为浮点数，否则除法运算为0
    count = 0
    while count < max_loop:  # 对每次循环求损失，并用测试集求预测准确率
        h = sigmoid(np.dot(array_x, theta))
        theta = theta - alpha * ((1.0 / m) * np.dot(array_x.T, (h - array_y)) + (my_Lambda / m) * np.r_[[[0]], theta[1:]])
        thetas.append(theta)
        cost = J(theta, array_x, array_y, int(my_Lambda))
        costs.append(cost)

        testPredict = predict(test_X, theta)
        accuracy = accuracy_rate(test_Y, testPredict)
        accuracies.append(accuracy)

        if abs(costs[-1] - costs[-2]) < epsilon:
            break
        count += 1
    return thetas, costs, count, accuracies


# 多分类时无需在每个迭代中求预测准确率，故另外定义一个梯度下降函数
def gradient2(array_x, array_y, alpha, epsilon, max_loop):
    m = len(array_x)
    n = 5
    # 初始化模型参数，n个特征对应n个参数
    theta = np.zeros((n, 1))
    cost = J(theta, array_x, array_y)  # 当前代价
    costs = [cost]
    thetas = [theta]
    my_lambda = float(0)
    count = 0
    while count < max_loop:
        h = sigmoid(np.dot(array_x, theta))
        theta = theta - alpha * ((1.0 / m) * np.dot(array_x.T, (h - array_y)) + (my_lambda / m) * np.r_[[[0]], theta[1:]])
        thetas.append(theta)
        cost = J(theta, array_x, array_y, int(my_lambda))
        costs.append(cost)
        if abs(costs[-1] - costs[-2]) < epsilon:
            break
        count += 1
    return thetas, costs, count
