import Classifier as c
import InduceC45 as c45
import sys
import pandas

res_file_name = 'decision_tree.json'

def generate_data_subsets(df, k):
    data_subsets = []
    data_len = df.shape[0]
    for i in range(0, k):
        print(max(df.shape[0],int(data_len/k)))
        print(len(df))
        subset = df.sample(n=max(df.shape[0],int(data_len/k)))
        print(subset)
        data_subsets.append(subset)
        df = df.drop(subset.index)
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
        test = data_subsets[i]
        training = get_training_dataset(i, data_subsets)

        # produces json
        c45.c45_produce_json(training, attributes, threshold, class_attr,
                            class_labels, attr_domain_dict, res_file_name)

        # classify and generate statistics
        matrix, stats = c.classifier(test.to_csv(), res_file_name)
        print(matrix)
        print(stats)

def all_but_one(df, attributes, threshold, class_attr, 
          class_labels, attr_domain_dict): 

    for i in range(len(df)):
        test = df.loc[i]
        train = df.drop(test)

        # produces json
        c45.c45_produce_json(train, attributes, threshold, class_attr,
                            class_labels, attr_domain_dict, res_file_name)

        # classify and generate statistics


if __name__ == "__main__":
    if len(sys.argv) == 4:
        training_file = sys.argv[1]
        restr_file = sys.argv[2]
        k = int(sys.argv[3])
    else:
        training_file = sys.argv[1]
        k = int(sys.argv[2])

    df = pandas.read_csv(training_file)
    class_attr = df.iloc[1][0]
    df = df.drop(labels=[0, 1])
    class_labels = df[class_attr].unique()
    attributes = [attr for attr in df.columns if not attr == class_attr]
    attr_domain_dict = c45.build_attr_domain_dict(df, attributes)
    threshold = 0
    
    if k > -1:
        data_subsets = generate_data_subsets(df, k)
        kfold(data_subsets, attributes, threshold, class_attr, class_labels, attr_domain_dict)

