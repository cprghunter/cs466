import pandas
import csv
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

    def add_child(self, edge_label, other, gt_le=None):
        if gt_le:
            self.edges[(gt_le, edge_label)] = other
        else:
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
            if type(key) == tuple: # continuous attr split
                edge_dict["edge"] = {"value": key[1], "direction": key[0]}
            else:
                edge_dict["edge"] = {"value": key}

            if isinstance(self.edges[key], Leaf):
                edge_dict["edge"]["leaf"] = self.edges[key].to_dict()
            else:
                edge_dict["edge"]["node"] = self.edges[key].to_dict()
            edge_list.append(edge_dict)
        json_dict["edges"] = edge_list
        return json_dict


def calculate_entropy(df, class_attr, class_labels, class_label_counts=None):
    entropy = 0
    if not class_label_counts:
        class_label_counts = df[class_attr].value_counts()
    for class_label in class_labels:
        if class_label in class_label_counts:
            prob = class_label_counts[class_label] / df.shape[0]
            if prob != 0:
                entropy += prob * math.log2(prob)
    return -1 * entropy

def calculate_attribute_entropy(df, class_attr, class_labels, attr_domain, attr, 
                                x=None):
    total = 0
    if x != None:
        greater_than_x = df[df[attr] >= x]
        less_than_x = df[df[attr] < x]
        total += (len(greater_than_x) * calculate_entropy(greater_than_x,
                                                          class_attr,
                                                          class_labels) / df.shape[0])
        total += (len(less_than_x) * calculate_entropy(less_than_x,
                                                          class_attr,
                                                          class_labels) / df.shape[0])
        return total
    attr_value_counts = df[attr].value_counts()
    for value in attr_domain:
        if value in attr_value_counts:
            total += (attr_value_counts[value]
                      * calculate_entropy(df.loc[df[attr] == value], 
                                            class_attr, class_labels) / df.shape[0])
    return total

def find_best_split(df, attribute, class_attr, class_labels, entropy_of_unsplit):
    sorted_df = df.sort_values(attribute, ignore_index=True)
    max_gain = None
    max_gain_row = None
    """
    counts = {label : [0 for _ in range(len(sorted_df))] for label in class_labels} 
    for idx, row in sorted_df.iterrows():
        if idx == 0:
            continue
        for class_label in class_labels:
            if row[class_attr] == class_label: 
                counts[class_label][idx] = counts[class_label][idx - 1] + 1
            else:
                counts[class_label][idx] = counts[class_label][idx - 1]
    for idx, row in sorted_df.iterrows():
        alpha_counts = {class_label: counts[class_label][idx] 
                                            for class_label in class_labels}
        #print(f"{attribute}: split on {row[attribute]}: {alpha_counts}")
        gain = entropy_of_unsplit - calculate_entropy(sorted_df, class_attr, 
                                            class_labels, class_label_counts=alpha_counts)
        if idx == 0:
            max_gain = gain
            max_gain_row = row
        if gain > max_gain:
            max_gain = gain
            max_gain_row = row
    """
    for idx, row in sorted_df.iterrows():
        gain = entropy_of_unsplit - calculate_attribute_entropy(sorted_df, class_attr, 
                                            class_labels, {}, attribute, x=row[attribute])
        if idx == 0:
            max_gain = gain
            max_gain_row = row
        if gain > max_gain:
            max_gain = gain
            max_gain_row = row
    return max_gain_row[attribute]
        
def select_splitting_attribute(df, attributes, class_attr, class_labels, attr_domain_dict, threshold):
    entropy_of_unsplit = calculate_entropy(df, class_attr, class_labels)
    gain = {}
    cont_split = {}
    x = None
    for attribute in attributes:
        if len(attr_domain_dict[attribute]):
            attr_entropy = calculate_attribute_entropy(df, class_attr, 
                                                        class_labels, 
                                                        attr_domain_dict[attribute],
                                                        attribute)
        else: # continuous attributes
            x = find_best_split(df, attribute, class_attr, class_labels, entropy_of_unsplit)
            attr_entropy = calculate_attribute_entropy(df, class_attr, 
                                                        class_labels, 
                                                        attr_domain_dict[attribute],
                                                        attribute, x=x)
            cont_split[attribute] = x

        info_gain = entropy_of_unsplit - attr_entropy
        gain[attribute] = info_gain

    max_gain = max(gain, key=gain.get)
    if not len(attr_domain_dict[max_gain]):
        x = cont_split[max_gain]
    else:
        x = None
    if gain[max_gain] > threshold:
        return max_gain, x
    return None, None

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

        best_split, x = select_splitting_attribute(df, attributes, class_attr, class_labels,
                                                attr_domain_dict, threshold)
        if not best_split: # case where all splits are below threshold
            return get_leaf_with_most_freq_class(df, class_attr)
        else:
            if x == None: # non-continuous attibutes
                node = Node(best_split)
                attributes.remove(best_split)
                for attr_value in attr_domain_dict[best_split]:
                    if not df.loc[df[best_split] == attr_value].shape[0]:
                        node.add_child(attr_value,
                                       get_leaf_with_most_freq_class(df, class_attr))
                    else:
                        node.add_child(attr_value, 
                                       c45(df.loc[df[best_split] == attr_value],
                                       copy.deepcopy(attributes), threshold, class_attr, class_labels,
                                       attr_domain_dict))
            else: # continuous attribute
                node = Node(best_split)
                greater_than_x = df[df[best_split] >= x]
                less_than_x = df[df[best_split] < x]
                node.add_child(x, c45(greater_than_x, copy.deepcopy(attributes),
                                      threshold, class_attr, class_labels,
                                      attr_domain_dict), gt_le="gt")
                node.add_child(x, c45(less_than_x, copy.deepcopy(attributes),
                                      threshold, class_attr, class_labels,
                                      attr_domain_dict), gt_le="le")

            return node

def build_attr_domain_dict(df, attributes, attr_domain_size):
    attr_domain = {}
    for attribute in attributes:
        if int(attr_domain_size[attribute]):
            attr_domain[attribute] = df[attribute].unique()
        else:
            attr_domain[attribute] = [] # numeric attributes
    return attr_domain

def c45_produce_json(df, attributes, threshold, class_attr, 
                    class_labels, attr_domain_dict, res_file):
    tree = c45(df, attributes, threshold, class_attr, class_labels, attr_domain_dict)
    tree_dict = tree.to_dict()
    out = {}
    out["dataset"] = sys.argv[1]
    if "decision" in tree_dict:
        out["leaf"] = tree_dict
    else:
        out["node"] = tree_dict
    with open(res_file, 'w') as f:
        f.write(json.dumps(out))

def c45_get_tree_dict(df, attributes, threshold, class_attr, 
        class_labels, attr_domain_dict):
    tree = c45(df, attributes, threshold, class_attr, class_labels, attr_domain_dict)
    tree_dict = tree.to_dict()
    out = {}
    out["dataset"] = sys.argv[1]
    if "decision" in tree_dict:
        out["leaf"] = tree_dict
    else:
        out["node"] = tree_dict
    return out

def preprocess_data(data_df, restr_file=None):
    class_attr = data_df.iloc[1][0]
    if restr_file:
        cols = list(data_df.columns.values)
        with open(sys.argv[2], 'r') as f:
            restr = csv.reader(f)
            i = 0
            for row in restr:
                for col in row:
                    if col == "0":
                        data_df = data_df.drop(labels=cols[i], axis=1)
                    i += 1
    attr_domain_size = {attr: data_df[attr].iloc[0] for attr in data_df.columns}  
    data_df = data_df.drop(labels=[0, 1])
    class_labels = data_df[class_attr].unique()

    attributes = [attr for attr in data_df.columns if not attr == class_attr]
    attr_domain_dict = build_attr_domain_dict(data_df, attributes, attr_domain_size)

    return data_df, class_attr, class_labels, attributes, attr_domain_dict

if __name__ == "__main__":
    data_df = pandas.read_csv(sys.argv[1], index_col=None)
    # data_df[col][0] is number of values in the attributes domain
    """
    class_attr = data_df.iloc[1][0]

    attr_domain_size = {attr: data_df[attr].iloc[0] for attr in data_df.columns}  
    data_df = data_df.drop(labels=[0, 1])
    class_attr_values = data_df[class_attr].unique()

    attributes = [attr for attr in data_df.columns if not attr == class_attr]
    attr_domain_dict = build_attr_domain_dict(data_df, attributes, attr_domain_size)
    """
    data_df, class_attr, class_labels, attributes, attr_domain_dict = preprocess_data(data_df)

    #print(f"class attr: {class_attr}, class labels {class_attr_values}")
    c45_produce_json(data_df, attributes, .3, class_attr, class_labels,
            attr_domain_dict, RES_FNAME)
