import pandas as pd
import numpy as np
import argparse
import sys
import json

def classifier(dataset, tree):

    data = pd.read_csv(dataset)
    df = pd.DataFrame(data)
    df = df.drop(labels=[0,1])
    open_tree = open(tree, 'r')
    dt = json.load(open_tree)
    outcomes = df.iloc[:, -1].unique()
    print(outcomes)






if __name__ == '__main__':
    classifier(sys.argv[1], sys.argv[2])