import Classifier as c
import Induce45 as c45
import sys
import pandas


def generate_data_subsets(df, restr_file, k):
    data_subsets = []
    data_len = df.shap[0]
    for i in range(k):
        subset = df.sample(n=max(df.shape[0],int(data_len/k)))
        data_subsets.append(subset)
        df.drop(subset.index)
    return data_subsets

def get_training_dataset(i, data_subsets):
    df = pandas.DataFrame()
    for j  in range(len(data_subsets)):
        if j != i:
            df.append(data_subsets[j])
    return df

def kfold(data_subsets, attributes, threshold, class_attr, 
          class_labels, attr_domain_dict):

    for i in range(len(data_subsets)):
        test = data_subset[i]
        training = get_training_dataset(i, data_subsets)

        # produces json
        c45.c45_produce_json(training, attributes, threshold, class_attr,
                            class_labels, attr_domain_dict)

        # classify and generate statistics

def all_but_one(df, attributes, threshold, class_attr, 
          class_labels, attr_domain_dict): 

    for i in range(len(df)):
        test = df.loc[i]
        df.drop(test)

        # produces json
        c45.c45_produce_json(df, attributes, threshold, class_attr,
                            class_labels, attr_domain_dict)

        # classify and generate statistics

if __name__ == "__main__":
    if sys.argc == 4:
        training_file = sys.argv[1]
        restr_file = sys.argv[2]
        k = int(sys.argv[3])
    else:
        training_file = sys.argv[1]
        k = int(sys.argv[2])

    df = pandas.read_csv(training_file)
    class_attr = data_df.iloc[1][0]
    data_df = data_df.drop(labels=[0, 1])
    class_attr_values = data_df[class_attr].unique()
    attributes = [attr for attr in df.columns if not attr == class_attr]
    attr_domain_dict = c45.build_attr_domain_dict(data_df, attributes)

