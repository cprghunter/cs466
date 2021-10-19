import Classifier as c
import sys


def generate_data_subsets(df, k):
    data_subsets = []
    data_len = df.shap[0]
    if k != -1:
        for i in range(k):
            subset = df.sample(n=max(df.shape[0],int(data_len/k)))
            data_subsets.append(subset)
            df.drop(subset.index)
    else:
        for i in range(data_len):
            continue


if __name__ == "__main__":
    if sys.argc == 4:
        training_file = sys.argv[1]
        restr_file = sys.argv[2]
        k = int(sys.argv[3])
    else:
        training_file = sys.argv[1]
        k = int(sys.argv[3])

