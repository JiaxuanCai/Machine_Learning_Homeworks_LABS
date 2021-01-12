import numpy as np
from collections import Counter
import pandas as pd
from tools.RegressionTreeTools import *


def training(dataset, feature_label=None):
    dataset = np.array(dataset)
    targets = dataset[:, -1]
    if np.unique(targets).shape[0] == 1:
        return targets[0]
    if dataset.shape[1] == 1:
        return Counter(targets.tolist()).most_common(1)[0]
    if feature_label is None:
        feature_label = [i for i in range(dataset.shape[1] - 1)]

    best_feature, best_value = choose_best_feature(dataset)
    best_feature_label = feature_label[best_feature]
    mytree = dict()
    mytree['FeatLabel'] = best_feature_label
    mytree['FeatValue'] = best_value
    l_set, r_set = split_dataset(dataset, best_feature, best_value)
    mytree['left'] = training(l_set, feature_label)
    mytree['right'] = training(r_set, feature_label)
    return mytree

def predict(tree, test_data, feature_label=None):
    if not isinstance(tree, dict):
        return tree
    if feature_label is None:
        feature_label = [i for i in range(test_data.shape[1] - 1)]

    best_feature_label = tree['FeatLabel']
    best_feature = feature_label.index(best_feature_label)
    if test_data[best_feature] <= tree['FeatValue']:
        return predict(tree['left'], test_data, feature_label)
    else:
        return predict(tree['right'], test_data, feature_label)


if __name__ == '__main__':
    student = np.delete(np.array(pd.read_csv('datasets/student_admit.csv')), 0, axis=1)
    print(training(student))

    housing = np.delete(np.array(pd.read_csv('datasets/boston_housing.csv')), 0, axis=1)
    print(training(housing))

