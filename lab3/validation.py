import Classifier as c
import InduceC45 as c45
import sys


def generate_data_subsets(df, restr_file, k):
    data_subsets = []
    data_len = df.shape[0]
    if k != -1:
        for i in range(k):
            subset = df.sample(n=max(df.shape[0],int(data_len/k)))
            data_subsets.append(subset)
            df.drop(subset.index)
            c45(df, restr_file)
            matrix, stats = Classifier(subset.to_csv(), 'decision_tree.json')
            print(matrix, stats)
            
    else:
        for i in range(data_len):
            continue

def validate(sets, k):
    for i in range(0, k):


if __name__ == "__main__":
    if len(sys.argv) == 4:
        training_file = sys.argv[1]
        restr_file = sys.argv[2]
        k = int(sys.argv[3])
    else:
        training_file = sys.argv[1]
        k = int(sys.argv[2])

