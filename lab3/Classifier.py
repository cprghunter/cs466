import pandas as pd
import numpy as np
import argparse
import sys
import json

def classifier(dataset, tree):
    data = pd.read_csv(dataset)
    df = pd.DataFrame(data)
    if df.iloc[1,0] != np.nan:
        use_column = df.iloc[1,0]
        df = df.drop(labels=[0,1])
        outcomes = df.loc[:, str(use_column)].unique()
    
    
    
    open_tree = open(tree, 'r')
    decision_tree = json.load(open_tree)
    
    print(outcomes)
    confusion_matrix = pd.DataFrame(0, index=outcomes, columns=outcomes)
    confusion_matrix.columns.name = "Actual \\ Classified"
    result_matrix = fill_matrix(confusion_matrix, decision_tree, df)
    print(result_matrix)

def fill_matrix(matrix, tree, data):
    for index, row in data.iterrows():
        actual = row[-1]
        predicted = traverse_tree(tree, row)
        matrix.at[actual, predicted] += 1
    return matrix

def traverse_tree(tree, row):
    if 'node' in tree.keys(): 
        find_edge = row[tree['node']['var']]
        for edge in tree['node']['edges']:
            if edge['edge']['value'] == find_edge:
                return traverse_tree(edge['edge'], row)
    else:
        return tree['leaf']['decision']


if __name__ == '__main__':
    classifier(sys.argv[1], sys.argv[2])