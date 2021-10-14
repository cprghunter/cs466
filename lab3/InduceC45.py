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

def calculate_entropy(df, class_attr, class_labels):
    entropy = 0
    class_label_counts = df[class_attr].value_counts()
    print(class_label_counts)
    for class_label in class_labels:
        if class_label in class_label_counts:
            prob = class_label_counts[class_label] / df.count
            entropy += prob * math.log2(prob)
    return -1 * entropy

def calculate_attribute_entropy(df, class_attr, class_labels, attr_domain, attr):
    total = 0
    attr_value_counts = df[attr].value_counts()
    for value in attr_domain:
        if value in attr_value_counts:
            total += (attr_value_counts[value]
                      * calculate_entropy(df.loc[df[attr] == value], 
                                            class_attr, class_labels) / df.count)
    return total
        
def select_splitting_attribute(df, class_attr, class_labels, attr_domain_dict, threshold):
    entropy_of_unsplit = calculate_entropy(df, class_attr, class_labels)
    gain = {}
    for attribute in attributes.keys():
        attr_entropy = calculate_attribute_entropy(df, class_attr, 
                                                    class_labels, 
                                                    attr_domain_dict[attribute],
                                                    attribute)
        info_gain = entropy_of_unsplit - attr_entropy
        gain[attribute] = info_gain
    max_gain = max(gain, key=gain.get)
    if gain[max_gain] > threshold:
        return max_gain
    return None

def c45(df, attributes, ifgain_threshold):
    if len(unique := df[class_attr].unique()) == 1:
        return Leaf(unique[0], 1) 
    else if not attributes:
        most_freq = df[class_attr].mode()
        # in the case of a tie for most frequent class attribute take first in 
        # list
        if not len(most_freq) == 1:
            most_freq = most_freq[:1]
        return Leaf(most_freq[0], len(df.loc[df[class_attr] == most_freq[0]]) / len(df))
    else:
        best_split = select_splitting_attribute(df, class_attr, class_labels,
                                                attr_domain_dict, threshold)
        if not best_split: # case where all splits are below threshold
            df[class_attr]
            return Leaf(

    """
    split:
        n  = Node()
        best_split = get_best_split_attr()
        for value in best_split:
            n.add_edge(value, c45(df, attributes - best_split, 
    """

def build_attr_domain_dict(df, attributes):
    attr_domain = {}
    for attribute in attributes:
        attr_doman[attribute] = df[attribute].unique()
    return attr_domain

data_df = pandas.read_csv(sys.argv[1], index_col=None)
# data_df[col][0] is number of values in the attributes domain
class_attr = data_df.iloc[1][0]
data_df = data_df.drop(labels=[0, 1])
class_attr_values = data_df[class_attr].unique()

attributes = [attr if not attr == class_attr for attr in data_df.columns]
attr_domain_dict = build_attr_domain_dict(data_df, attributes)

print(class_attr_values)
tree = c45(data_df, [], 0)
print(tree)
print(c45(data_df.loc[data_df[class_attr] == class_attr_values[0]], [], 0))
