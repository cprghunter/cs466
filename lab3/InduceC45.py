import pandas
import copy
import sys
import math
import json

RES_FNAME = "decision_tree.json"

class Leaf:
    def __init__(self, class_label, percent):
        self.class_label = class_label
        self.percent = percent
        self.parent = None

    def __repr__(self):
        return f"class label: {self.class_label}, percent: {self.percent}"

    def to_dict(self):
        json_dict = {}
        json_dict["decision"] = self.class_label
        json_dict["p"] = self.percent
        return json_dict


class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.parent = None
        self.edges = {}

    def add_child(self, edge_label, other):
        self.edges[edge_label] = other
        other.parent = self

    def __repr__(self):
        return f"node: {self.attribute}, edges: {self.edges}"
    
    def to_dict(self):
        json_dict = {}
        json_dict["var"] = self.attribute
        edge_list = []
        for key in self.edges:
            edge_dict = {}
            edge_dict["edge"] = {"value": key}
            if isinstance(self.edges[key], Leaf):
                edge_dict["edge"]["leaf"] = self.edges[key].to_dict()
            else:
                edge_dict["edge"]["node"] = self.edges[key].to_dict()
            edge_list.append(edge_dict)
        json_dict["edges"] = edge_list
        return json_dict


def calculate_entropy(df, class_attr, class_labels):
    entropy = 0
    class_label_counts = df[class_attr].value_counts()
    for class_label in class_labels:
        if class_label in class_label_counts:
            prob = class_label_counts[class_label] / df.shape[0]
            entropy += prob * math.log2(prob)
    return -1 * entropy

def calculate_attribute_entropy(df, class_attr, class_labels, attr_domain, attr):
    total = 0
    attr_value_counts = df[attr].value_counts()
    for value in attr_domain:
        if value in attr_value_counts:
            total += (attr_value_counts[value]
                      * calculate_entropy(df.loc[df[attr] == value], 
                                            class_attr, class_labels) / df.shape[0])
    return total
        
def select_splitting_attribute(df, attributes, class_attr, class_labels, attr_domain_dict, threshold):
    entropy_of_unsplit = calculate_entropy(df, class_attr, class_labels)
    gain = {}
    for attribute in attributes:
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

def get_leaf_with_most_freq_class(df, class_attr):
    most_freq = df[class_attr].mode()
    return Leaf(most_freq[0], len(df.loc[df[class_attr] == most_freq[0]]) / len(df))


def c45(df, attributes, threshold, class_attr, class_labels, attr_domain_dict):
    if len(unique := df[class_attr].unique()) == 1:
        return Leaf(unique[0], 1) 
    elif not attributes:
        # in the case of a tie for most frequent class attribute take first in 
        # list
        return get_leaf_with_most_freq_class(df, class_attr)
    else:

        best_split = select_splitting_attribute(df, attributes, class_attr, class_labels,
                                                attr_domain_dict, threshold)
        if not best_split: # case where all splits are below threshold
            return get_leaf_with_most_freq_class(df, class_attr)
        else:
            node = Node(best_split)
            attributes.remove(best_split)
            #for attr_value in df[best_split].unique(): replace with with attr_domain_dict look
            for attr_value in attr_domain_dict[best_split]:
                if not df.loc[df[best_split] == attr_value].shape[0]:
                    node.add_child(attr_value,
                                   get_leaf_with_most_freq_class(df, class_attr))
                else:
                    node.add_child(attr_value, 
                                   c45(df.loc[df[best_split] == attr_value],
                                   copy.deepcopy(attributes), threshold, class_attr, class_labels,
                                   attr_domain_dict))
            return node

def build_attr_domain_dict(df, attributes):
    attr_domain = {}
    for attribute in attributes:
        attr_domain[attribute] = df[attribute].unique()
    return attr_domain

def c45_produce_json(df, attributes, threshold, class_attr, 
                    class_labels, attr_domain_dict, res_file):
    tree = c45(df, attributes, threshold, class_attr, class_labels, attr_domain_dict)
    tree_dict = {}
    tree_dict["dataset"] = sys.argv[1]
    tree_dict["node"] = tree.to_dict()
    with open(res_file, 'w') as f:
        f.write(json.dumps(tree_dict))


if __name__ == "__main__":
    data_df = pandas.read_csv(sys.argv[1], index_col=None)
    # data_df[col][0] is number of values in the attributes domain
    class_attr = data_df.iloc[1][0]
    data_df = data_df.drop(labels=[0, 1])
    class_attr_values = data_df[class_attr].unique()

    attributes = [attr for attr in data_df.columns if not attr == class_attr]
    attr_domain_dict = build_attr_domain_dict(data_df, attributes)

    print(f"class labels {class_attr_values}")
    c45_produce_json(data_df, attributes, 0, class_attr, class_attr_values,
            attr_domain_dict, RES_FNAME)
