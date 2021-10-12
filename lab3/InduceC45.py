import pandas
import sys


class Node:
    def __init__(self, attribute):
        self.adjacent = {}

    def build_adjacencies(self):
        for edge in self.edges:
            self.adjacent[edge] = Node(["cat"])

def c45(dataframe, attributes, tree, ifgain_threshold):
    if not attributes:
        most_freq = dataframe[class_attr].mode()
        # in the case of a tie for most frequent class attribute take first in 
        # list
        if not len(most_freq) == 1:
            most_freq = most_freq[:1]
        tree.add_neighbor(most_freq[0])

data_df = pandas.read_csv(sys.argv[1], index_col=None)
# data_df[col][0] is number of values in the attributes domain
class_attr = data_df.iloc[1][0]
class_attr_values = data_df[class_attr].unique()
tree = c45(data_df, data_df.columns, 
print(data_df)
print(class_attr)
