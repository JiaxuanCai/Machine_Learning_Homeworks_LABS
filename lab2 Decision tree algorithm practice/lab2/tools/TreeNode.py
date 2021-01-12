class TreeNode:  # 表示决策树节点的类
    node_index = 0
    parent = None
    property_name = None
    children = None
    verdiction = None
    split = None
    data_index = None
    attr_value = None
    rest_attribute = None

    def __init__(self, parent=None, property_name=None, children=None, verdiction=None, split=None, data_index=None, attr_value=None, rest_attribute=None):
        self.parent = parent
        self.attribute_name = property_name
        self.attribute_value = attr_value
        self.children = children
        self.verdiction = verdiction
        self.split = split
        self.data_index = data_index
        self.index = TreeNode.node_index
        self.rest_attribute = rest_attribute
        TreeNode.node_index += 1

    def to_string(self):  # 以字符串形式输出当前节点信息
        this_string = '当前节点编号为 : ' + str(self.index) + ";\n"
        if not (self.parent is None):  # 如果不是根节点
            parent_node = self.parent
            this_string = this_string + '父节点编号 : ' + str(parent_node.index) + ";\n"
            this_string = this_string + str(parent_node.attribute_name) + " : " + str(self.attribute_value) + ";\n"
        this_string = this_string + "节点数据 : " + str(self.data_index) + ";\n"
        if not (self.children is None):  # 如果不是叶子节点
            this_string = this_string + '划分属性为 ' + str(self.attribute_name) + ";\n"
            child_list = []
            for child in self.children:
                child_list.append(child.index)
            this_string = this_string + '子节点 : ' + str(child_list)
        if not (self.verdiction is None):  # 如果是叶节点
            this_string = this_string + '叶节点分类结果 : ' + self.verdiction
        return this_string