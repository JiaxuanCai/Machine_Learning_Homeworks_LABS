import numpy as np


def gini_index_tst(self, y):
    y_len = len(y)
    # 统计各个y值在整个y中出现的次数，方便后面套基尼系数公式
    y_counter = {}
    for i in range(0, y_len):
        y_value = y[i]
        if y_counter.get(y_value) is None:
            y_counter[y_value] = 0
        y_counter[y_value] = y_counter[y_value] + 1

    gini_index_value = 1.0
    for k, counter_num in y_counter.items():
        p_k = 1.0 * counter_num / y_len
        gini_index_value = gini_index_value - p_k * p_k
    return gini_index_value


def gini_index_gain(self, y_sub, y):
    y = np.array(y)
    y_sub = np.array(y_sub)

    property_gini_index_value = 0.0
    for prop_value in set(y_sub):
        props = np.where(y_sub == prop_value)
        partial_y_sub = y_sub[props]
        partial_y = y[props]
        p_k = 1.0 * len(partial_y_sub) / len(y_sub)
        property_gini_index_value += p_k * self.gini_index(partial_y)
    return self.gini_index(y) - property_gini_index_value

def gini(labels=[]):
    """
    计算数据集的基尼值
    :param labels: 数据集的类别标签
    :return:
    """
    data_count = {}
    for label in labels:
        if data_count.__contains__(label):
            data_count[label] += 1
        else:
            data_count[label] = 1

    n = len(labels)
    if n == 0:
        return 0

    gini_value = 1
    for key, value in data_count.items():
        gini_value = gini_value - (value / n) * (value / n)

    return gini_value


def gini_index_basic(n, attr_labels={}):
    gini_value = 0
    for attribute, labels in attr_labels.items():
        gini_value = gini_value + len(labels) / n * gini(labels)

    return gini_value


def gini_index(attributes=[], labels=[], is_value=False):
    """
    计算一个属性的基尼指数
    :param attributes: 当前数据在该属性上的属性值列表
    :param labels:
    :param is_value:
    :return:
    """
    n = len(labels)
    attr_labels = {}
    gini_value = 0  # 最终要返回的结果
    split = None  #

    if is_value:  # 属性值是连续的数值
        sorted_attributes = attributes.copy()
        sorted_attributes.sort()
        split_points = []
        for i in range(0, n - 1):
            split_points.append((sorted_attributes[i + 1] + sorted_attributes[i]) / 2)

        split_evaluation = []
        for current_split in split_points:
            low_labels = []
            up_labels = []
            for i in range(0, n):
                if attributes[i] <= current_split:
                    low_labels.append(labels[i])
                else:
                    up_labels.append(labels[i])
            attr_labels = {'small': low_labels, 'large': up_labels}
            split_evaluation.append(gini_index_basic(n, attr_labels=attr_labels))

        gini_value = min(split_evaluation)
        split = split_points[split_evaluation.index(gini_value)]

    else:  # 属性值是离散的词
        for i in range(0, n):
            if attr_labels.__contains__(attributes[i]):
                temp_list = attr_labels[attributes[i]]
                temp_list.append(labels[i])
            else:
                temp_list = []
                temp_list.append(labels[i])
                attr_labels[attributes[i]] = temp_list

        gini_value = gini_index_basic(n, attr_labels=attr_labels)

    return gini_value, split
