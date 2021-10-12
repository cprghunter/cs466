import pandas
import sys
import math

class Leaf:
    def __init__(self, class_label, percent):
        self.class_label = class_label
        self.percent = percent

    def __repr__(self):
        return f"class label: {self.class_label}, percent: {self.percent}"

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}

    def add_node_child(self, edge_label, other):
        self.adjacent[edge_label] = other
        other.parent = self

    def add_leaf_child(self, other, edge_label):
        self.adjacent[edge_label] = other
        other.parent = self

def c45(df, attributes, ifgain_threshold):
    if len(unique := df[class_attr].unique()) == 1:
        return Leaf(unique[0], 1) 

    if not attributes:
        most_freq = df[class_attr].mode()
        # in the case of a tie for most frequent class attribute take first in 
        # list
        if not len(most_freq) == 1:
            most_freq = most_freq[:1]
        return Leaf(most_freq[0], len(df.loc[df[class_attr] == most_freq[0]]) / len(df))

    """
    split:
        n  = Node()
        best_split = get_best_split_attr()
        for value in best_split:
            n.add_edge(value, c45(df, attributes - best_split, 
    """



data_df = pandas.read_csv(sys.argv[1], index_col=None)
# data_df[col][0] is number of values in the attributes domain
class_attr = data_df.iloc[1][0]
data_df = data_df.drop(labels=[0, 1])
class_attr_values = data_df[class_attr].unique()
print(class_attr_values)
tree = c45(data_df, [], 0)
print(tree)
print(c45(data_df.loc[data_df[class_attr] == class_attr_values[0]], [], 0))
