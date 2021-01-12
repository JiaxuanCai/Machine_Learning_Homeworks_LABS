from tools.gini_index import *
from tools.TreeNode import *

def continuous(s): # 判断数据属性是否连续
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def finish_node(current_node=TreeNode, data=[], label=[]):
    one_class = True
    this_data_index = current_node.data_index

    for i in this_data_index:
        for j in this_data_index:
            if label[i] != label[j]:
                one_class = False
                break
        if not one_class:
            break
    if one_class:
        current_node.verdiction = label[this_data_index[0]]
        return

    rest_title = current_node.rest_attribute
    if len(rest_title) == 0:
        label_count = {}
        temp_data = current_node.data_index
        for index in temp_data:
            if label_count.__contains__(label[index]):
                label_count[label[index]] += 1
            else:
                label_count[label[index]] = 1
        final_label = max(label_count)
        current_node.verdiction = final_label
        return

    title_gini = {}
    title_split_value = {}
    for title in rest_title:
        attr_values = []
        current_label = []
        for index in current_node.data_index:
            this_data = data[index]
            attr_values.append(this_data[title])
            current_label.append(label[index])
        temp_data = data[0]
        this_gain, this_split_value = gini_index(attr_values, current_label,
                                                 continuous(temp_data[title]))
        title_gini[title] = this_gain
        title_split_value[title] = this_split_value

    best_attr = min(title_gini, key=title_gini.get)
    current_node.attribute_name = best_attr
    current_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if continuous(a_data[best_attr]):
        split_value = title_split_value[best_attr]
        small_data = []
        large_data = []
        for index in current_node.data_index:
            this_data = data[index]
            if this_data[best_attr] <= split_value:
                small_data.append(index)
            else:
                large_data.append(index)
        small_str = '<=' + str(split_value)
        large_str = '>' + str(split_value)
        small_child = TreeNode(parent=current_node, data_index=small_data, attr_value=small_str,
                                        rest_attribute=rest_title.copy())
        large_child = TreeNode(parent=current_node, data_index=large_data, attr_value=large_str,
                                        rest_attribute=rest_title.copy())
        current_node.children = [small_child, large_child]
    else:
        best_titlevalue_dict = {}
        for index in current_node.data_index:
            this_data = data[index]
            if best_titlevalue_dict.__contains__(this_data[best_attr]):
                temp_list = best_titlevalue_dict[this_data[best_attr]]
                temp_list.append(index)
            else:
                temp_list = [index]
                best_titlevalue_dict[this_data[best_attr]] = temp_list

        children_list = []
        for key, index_list in best_titlevalue_dict.items():
            a_child = TreeNode(parent=current_node, data_index=index_list, attr_value=key,
                                        rest_attribute=rest_title.copy())
            children_list.append(a_child)
        current_node.children = children_list
    for child in current_node.children:
        finish_node(child, data, label)


def finish_node_pre(current_node=TreeNode(), data=[], label=[], test_data=[], test_label=[]):
    one_class = True
    this_data_index = current_node.data_index

    for i in this_data_index:
        for j in this_data_index:
            if label[i] != label[j]:
                one_class = False
                break
        if not one_class:
            break
    if one_class:
        current_node.verdiction = label[this_data_index[0]]
        return
    rest_title = current_node.rest_attribute
    if len(rest_title) == 0:
        label_count = {}
        temp_data = current_node.data_index
        for index in temp_data:
            if label_count.__contains__(label[index]):
                label_count[label[index]] += 1
            else:
                label_count[label[index]] = 1
        final_label = max(label_count)
        current_node.verdiction = final_label
        return
    data_count = {}
    for index in current_node.data_index:
        if data_count.__contains__(label[index]):
            data_count[label[index]] += 1
        else:
            data_count[label[index]] = 1
    before_judge = max(data_count, key=data_count.get)
    current_node.verdiction = before_judge
    before_accuracy = current_accuracy(current_node, test_data, test_label)
    title_gini = {}
    title_split_value = {}
    for title in rest_title:
        attr_values = []
        current_label = []
        for index in current_node.data_index:
            this_data = data[index]
            attr_values.append(this_data[title])
            current_label.append(label[index])
        temp_data = data[0]
        this_gain, this_split_value = gini_index(attr_values, current_label,
                                                      continuous(temp_data[title]))
        title_gini[title] = this_gain
        title_split_value[title] = this_split_value

    best_attr = min(title_gini, key=title_gini.get)
    current_node.attribute_name = best_attr
    current_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if continuous(a_data[best_attr]):
        split_value = title_split_value[best_attr]
        small_data = []
        large_data = []
        for index in current_node.data_index:
            this_data = data[index]
            if this_data[best_attr] <= split_value:
                small_data.append(index)
            else:
                large_data.append(index)
        small_str = '<=' + str(split_value)
        large_str = '>' + str(split_value)
        small_child = TreeNode(parent=current_node, data_index=small_data, attr_value=small_str, rest_attribute=rest_title.copy())
        large_child = TreeNode(parent=current_node, data_index=large_data, attr_value=large_str, rest_attribute=rest_title.copy())
        small_data_count = {}
        for index in small_child.data_index:
            if small_data_count.__contains__(label[index]):
                small_data_count[label[index]] += 1
            else:
                small_data_count[label[index]] = 1
        small_child_judge = max(small_data_count, key=small_data_count.get)
        small_child.verdiction = small_child_judge
        large_data_count = {}
        for index in large_child.data_index:
            if large_data_count.__contains__(label[index]):
                large_data_count[label[index]] += 1
            else:
                large_data_count[label[index]] = 1
        large_child_judge = max(large_data_count, key=large_data_count.get)
        large_child.verdiction = large_child_judge  # 临时添加的一个判断
        current_node.children = [small_child, large_child]
    else:
        best_titlevalue_dict = {}
        for index in current_node.data_index:
            this_data = data[index]
            if best_titlevalue_dict.__contains__(this_data[best_attr]):
                temp_list = best_titlevalue_dict[this_data[best_attr]]
                temp_list.append(index)
            else:
                temp_list = [index]
                best_titlevalue_dict[this_data[best_attr]] = temp_list

        children_list = []
        for key, index_list in best_titlevalue_dict.items():
            a_child = TreeNode.TreeNode(parent=current_node, data_index=index_list, attr_value=key,
                                        rest_attribute=rest_title.copy())
            temp_data_count = {}
            for index in index_list:
                if temp_data_count.__contains__(label[index]):
                    temp_data_count[label[index]] += 1
                else:
                    temp_data_count[label[index]] = 1
            temp_child_judge = max(temp_data_count, key=temp_data_count.get)
            a_child.verdiction = temp_child_judge
            children_list.append(a_child)
        current_node.children = children_list

    current_node.verdiction = None
    later_accuracy = current_accuracy(current_node, test_data, test_label)
    if before_accuracy > later_accuracy:
        current_node.children = None
        current_node.verdiction = before_judge
        return
    else:
        for child in current_node.children:
            finish_node_pre(child, data, label, test_data, test_label)


def cart_tree(Data, title, label):
    n = len(Data)
    root_data = []
    for i in range(0, n):
        root_data.append(i)
    root_node = TreeNode(data_index=root_data, rest_attribute=title.copy())
    finish_node(root_node, Data, label)
    return root_node

def precut_cart_tree(Data, title, label, test_data, test_label):
    n = len(Data)
    root_data = []
    for i in range(0, n):
        root_data.append(i)

    root_node = TreeNode(data_index=root_data, rest_attribute=title.copy())
    finish_node_pre(root_node, Data, label, test_data, test_label)
    return root_node

def post_pruning(decision_tree, test_data=[], test_label=[], train_label=[]):
    leaf_father = []  # 所有的孩子都是叶结点的结点集合
    iteration_list = []
    iteration_list.append(decision_tree)
    while len(iteration_list) > 0:
        current_node = iteration_list[0]
        children = current_node.children
        wanted = True  # 判断当前结点是否满足所有的子结点都是叶子结点
        if not (children is None):
            for child in children:
                iteration_list.append(child)
                temp_bool = (child.children is None)
                wanted = (wanted and temp_bool)
        else:
            wanted = False

        if wanted:
            leaf_father.append(current_node)
        iteration_list.remove(current_node)

    while len(leaf_father) > 0:
        # 如果叶父结点为空，则剪枝完成。对于不需要进行剪枝操作的叶父结点，我们也之间将其从leaf_father中删除
        current_node = leaf_father.pop()
        # 不进行剪枝在测试集上的正确率
        before_accuracy = current_accuracy(root_node=decision_tree, test_data=test_data, test_label=test_label)

        data_index = current_node.data_index
        label_count = {}
        for index in data_index:
            if label_count.__contains__(index):
                label_count[train_label[index]] += 1
            else:
                label_count[train_label[index]] = 1
        current_node.verdiction = max(label_count, key=label_count.get)  # 如果进行剪枝当前结点应该做出的判断
        later_accuracy = current_accuracy(root_node=decision_tree, test_data=test_data, test_label=test_label)

        if before_accuracy > later_accuracy:  # 不进行剪枝
            current_node.verdiction = None
        else:  # 进行剪枝
            current_node.children = None
            # 还需要检查是否需要对它的父节点进行判断
            parent_node = current_node.parent
            if not (parent_node is None):
                children_list = parent_node.children
                temp_bool = True
                for child in children_list:
                    if not (child.children is None):
                        temp_bool = False
                        break
                if temp_bool:
                    leaf_father.append(parent_node)


def print_tree(root=TreeNode):
    node_list = [root]
    while (len(node_list) > 0):
        current_node = node_list[0]
        print('--------------------------------------------')
        print(current_node.to_string())
        print('--------------------------------------------')
        children_list = current_node.children
        if not (children_list is None):
            for child in children_list:
                node_list.append(child)
        node_list.remove(current_node)


def classify_data(decision_tree=TreeNode, x={}):
    current_node = decision_tree
    while current_node.verdiction is None:
        if current_node.split is None:
            can_judge = False
            for child in current_node.children:
                if child.attribute_value == x[current_node.attribute_name]:
                    current_node = child
                    can_judge = True
                    break
            if not can_judge:
                return None
        else:
            child_list = current_node.children
            if x[current_node.attribute_name] <= current_node.split:
                current_node = child_list[0]
            else:
                current_node = child_list[1]

    return current_node.verdiction


def current_accuracy(root_node=TreeNode(), test_data=[], test_label=[]):
    accuracy = 0
    for i in range(0, len(test_label)):
        this_label = classify_data(root_node, test_data[i])
        if this_label == test_label[i]:
            accuracy += 1
    return accuracy / len(test_label)
