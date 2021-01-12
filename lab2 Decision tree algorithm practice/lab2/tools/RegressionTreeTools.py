import numpy as np

def cal_se(dataset):
    return np.var(dataset[:, -1]) * dataset.shape[0] if dataset.shape[0] > 0 else 0

def split_dataset(dataset, feature, value):
    left = dataset[np.nonzero(dataset[:, feature] <= value)[0], :]
    right = dataset[np.nonzero(dataset[:, feature] > value)[0], :]
    return left, right

def cal_split_se(dataset, feature):
    values = np.unique(dataset[:, feature])
    min_se, min_value = np.inf, 0
    for value in values:
        left, right = split_dataset(dataset, feature, value)
        new_se = cal_se(left) + cal_se(right)
        if new_se < min_se:
            min_se = new_se
            min_value = value
    return min_se, min_value

def choose_best_feature(dataset):
    m, n = dataset.shape[0], dataset.shape[1] - 1
    delta_gini, delta_info = np.inf, -np.inf
    best_feature, best_value = -1, 0
    for feature in range(n):
        new_se, value = cal_split_se(dataset, feature)
        if new_se < delta_gini:
            delta_gini = new_se
            best_value = value
            best_feature = feature
    return best_feature, best_value

def prune(self, tree, test_data):
    def istree(tr):
        return isinstance(tr, dict)
    def getmean(tr):
        if istree(tr['left']):
            tr['left'] = getmean(tr['left'])
        if istree(tr['right']):
            tr['right'] = getmean(tr['right'])
        return (tr['left'] + tr['right']) / 2
    left = right = None
    if not test_data:
        return getmean(tree)
    if istree(tree['left']) or istree(tree['right']):
        left, right = self.split_dataset(test_data, tree['FeatLabel'], tree['FeatValue'])
    if istree(tree['left']):
        tree['left'] = self.prune(tree['left'], left)
    if istree(tree['right']):
        tree['right'] = self.prune(tree['right'], right)
    if not istree(tree['left']) and not istree(tree['right']):
        left, right = self.split_dataset(test_data, tree['FeatLabel'], tree['FeatValue'])
        error_nomerge = np.sum(np.power(left[:, -1] - tree['left'], 2)) + \
                                np.sum(np.power(right[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge <= error_nomerge:
            return tree_mean
        else:
            return tree
    return tree