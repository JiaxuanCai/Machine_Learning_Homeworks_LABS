import csv
from tools.ClassificationTreeTools import *
from time import *

def run_test_pre():  # 前剪枝
    f = open('datasets/zoo.csv', 'r')
    csvreader = csv.reader(f)
    final_list = list(csvreader)
    final_list_copy = final_list.copy()
    final_list = [t[1:17] for t in final_list]
    final_list[1:102] = [[float(num) for num in item] for item in final_list[1:102]]
    title = final_list[0]
    train_datas = final_list[1:81]
    train_label = [t[17] for t in final_list_copy][1:81]

    test_datas = final_list[81:101]
    test_label = [t[17] for t in final_list_copy][81:101]
    train_data = []
    test_data = []
    for data_records in train_datas:
        a_dict = {}
        dim = len(data_records) - 1
        for i in range(0, dim+1):
            a_dict[title[i]] = data_records[i]
        train_data.append(a_dict)
        # train_label.append(data_records[dim])
    for data_records in test_datas:
        a_dict = {}
        dim = len(data_records) - 1
        for i in range(0, dim+1):
            a_dict[title[i]] = data_records[i]
        test_data.append(a_dict)
        # test_label.append(data_records[dim])

    begin_time = time()
    decision_tree = precut_cart_tree(train_data, title, train_label, test_data, test_label)
    end_time = time()
    run_time = end_time - begin_time
    print('运行时间：', run_time)

    print('训练的决策树是:')
    print_tree(decision_tree)
    print('\n')

    test_judge = []
    for data_records in test_data:
        test_judge.append(classify_data(decision_tree, data_records))
    print('决策树在测试数据集上的分类结果是：', test_judge)
    print('测试数据集的正确类别信息应该是：  ', test_label)

    accuracy = 0
    for i in range(0, len(test_label)):
        if test_label[i] == test_judge[i]:
            accuracy += 1
    accuracy /= len(test_label)
    print('决策树在测试数据集上的分类正确率为：' + str(accuracy * 100) + "%")

if __name__ == '__main__':
    run_test_pre()