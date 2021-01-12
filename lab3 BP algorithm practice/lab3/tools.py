import numpy as np


# sigmoid 函数的定义
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid函数的导数，其中out是隐藏层的输出（激励），即sigmoid函数本身的输出。
# np.multiply函数的作用是矩阵对应的元素分别相乘。
def derivative_sigmoid(out):
    return np.multiply(out, (1-out))


# 初始化权值矩阵，按照PPT中的方案初始化为-1/2到1/2中的随机值。
def set_initial_weights(num_layer, num_nodes, num_input, num_output):
    # 隐藏层的节点个数
    layerNum = [num_nodes for i in range(num_layer)]
    # 各层的节点个数（分别是输出层 隐藏层 输出层）
    layersNums = [num_input] + layerNum + [num_output]
    # 代表各层权值的矩阵
    Ws = []
    for idx, unit in enumerate(layersNums):
        if idx == len(layersNums) - 1:
            break
        nextUnit = layersNums[idx + 1]
        w = np.random.rand(nextUnit, unit + 1) - 0.5
        Ws.append(w)
    return Ws


# 带入报告中我推导出的代价函数，运算代价函数。
def compute_cost(thetas, y, regularization_index, a=None):
    # 样本条目数
    num_record = y.shape[0]
    # 训练完成后整个网络与真实值之差
    # 用矩阵的转置进行乘法运算
    error = -np.sum(np.multiply(y.T, np.log(a[-1])) + np.multiply((1 - y).T, np.log(1 - a[-1])))
    # 正则化项
    reg = np.sum([np.sum(np.multiply(theta[:, 1:], theta[:, 1:])) for theta in thetas])
    # 代入推导出的J公式，返回即可。
    return (1.0 / num_record) * error + (1.0 / (2 * num_record)) * regularization_index * reg


# 此函数的作用说明：
# 矩阵做运算时需要按标签类型的个数进行向量化，而输入数据只有一列。
# 此函数对y标签矩阵进行向量化
def vectorize_labels(y):
    # 使用np.ravel()可以影响原始矩阵
    labels = set(np.ravel(y))
    num_labels_class = len(labels)
    # 标签的最小值
    mini_label = min(labels)
    # 向量化之后的标签矩阵
    vector_labels = np.zeros((y.shape[0], num_labels_class), np.float64)
    for row, label in enumerate(y):
        vector_labels[row, label - mini_label] = 1.0
    return vector_labels


# 计算输出（激励值）
def compute_output(weights, X):
    # 层数
    mat_num_layer = list(range(len(weights) + 1))
    num_layer = len(mat_num_layer)
    outputs = list(range(num_layer))
    # 前向传播
    for layer in mat_num_layer:
        # 输入层
        if layer == 0:
            outputs[layer] = X.T
        # 隐藏层与输出层
        else:
            z = weights[layer - 1] * outputs[layer - 1]
            outputs[layer] = sigmoid(z)
        # 输入层与隐藏层添加偏置
        if layer != num_layer - 1:
            outputs[layer] = np.concatenate((np.ones((1, outputs[layer].shape[1])), outputs[layer]))
    return outputs


# 反向传播，计算梯度表达式
def gradient(thetas, a, y, regularization_index):
    num_record = y.shape[0]
    mat_layers = list(range(len(thetas) + 1))
    num_layers = len(mat_layers)
    d = list(range(len(mat_layers)))
    delta = [np.zeros(theta.shape) for theta in thetas]
    for layer in mat_layers[::-1]:
        if layer == 0:
            # 输入层不计算误差
            break
        if layer == num_layers - 1:
            # 输出层误差
            d[layer] = a[layer] - y.T
        else:
            # 忽略偏置
            d[layer] = np.multiply((thetas[layer][:, 1:].T * d[layer + 1]), derivative_sigmoid(a[layer][1:, :]))

    for l in mat_layers[0:num_layers - 1]:
        delta[l] += d[l + 1] * (a[l].T)
    D = [np.zeros(theta.shape) for theta in thetas]
    for l in range(len(thetas)):
        theta = thetas[l]
        # 偏置更新增量
        D[l][:, 0] = (1.0 / num_record) * (delta[l][0:, 0].reshape(1, -1))
        # 权值更新增量
        D[l][:, 1:] = (1.0 / num_record) * (delta[l][0:, 1:] + regularization_index * theta[:, 1:])
    return D


# 使用梯度下降法更新权重矩阵
def gradientDescent(thetas, X, y, alpha, regularization_index):
    # 样本数，特征数
    m, n = X.shape
    # 前向传播计算各个神经元的激活值
    a = compute_output(thetas, X)
    # 反向传播计算梯度增量
    D = gradient(thetas, a, y, regularization_index)
    # 计算预测代价
    J = compute_cost(thetas,y,regularization_index,a=a)
    # 更新权值
    for l in range(len(thetas)):
        thetas[l] = thetas[l] - alpha * D[l]
    if np.isnan(J):
        J = np.inf
    return J, thetas


def train(X, y, num_layer=1, num_nodes=8, alpha=1, regularization_index=0, iters=0):
    # 记录条数、输入维度数
    num_records, num_input = X.shape
    y = vectorize_labels(y)
    classNum = y.shape[1]
    weights = set_initial_weights(num_input=num_input, num_layer=num_layer, num_nodes=num_nodes, num_output=classNum)
    for i in range(iters):
        error, weights = gradientDescent(weights, X, y, alpha=alpha, regularization_index=regularization_index)
    return weights