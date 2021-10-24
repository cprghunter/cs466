import pandas as pd
import numpy as np
import sys

def knn(dataset, k, test_data):
    dataset = pd.read_csv(dataset)
    test_data = pd.read_csv(test_data)
    print(dataset, test_data)
    return

if __name__ == '__main__':
    dataset = sys.argv[1]
    k = int(sys.argv[2])
    test_data = sys.argv[3]
    knn(dataset, k, test_data)