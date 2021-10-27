import InduceC45 as c45
import Classifier as classify
import validation as validate
import argparse
import random
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--numattributes", "-m")
parser.add_argument("--numdataset", "-k")
parser.add_argument("--numtrees", "-n")
parser.add_argument("--dataset", "-d")

def get_random_data_subset(df, k):
    if k > df.shape[0]:
        raise ValueError("k must be smaller than size of dataset")
    return df.sample(n=k)

def select_random_attributes(attributes, attr_domain_dict, m):
    selected_attributes = random.choices(attributes, k=m)
    new_domain_dict = {attr: attr_domain_dict[attr] for attr in selected_attributes}
    return selected_attributes, new_domain_dict

def generate_forest(df, class_attr, class_labels, attributes, attr_domain_dict, threshold, k, m, n, gr=False):
    forest = []
    for _ in range(n):
        selected_df = get_random_data_subset(df, k)
        selected_attr, selected_attr_domain_dict = select_random_attributes(attributes,
                                                             attr_domain_dict,
                                                             m)
        forest.append(c45.c45_get_tree_dict(selected_df, selected_attr, threshold, class_attr, 
                                            class_labels, selected_attr_domain_dict, gr=gr))
    return forest

def get_forest_prediction(forest, data_point, class_labels):
    pred_count = {label: 0 for label in class_labels}
    for tree in forest:
        pred_count[classify.traverse_tree(tree, data_point)] += 1
    return max(pred_count, key=pred_count.get)

def predict_on_dataset(df, forest, class_attr, class_labels):
    confusion_matrix = pandas.DataFrame(0, index=class_labels, columns=class_labels)
    confusion_matrix.columns.name = "Actual \\ Classified"
    for idx, row in df.iterrows():
        predicted = get_forest_prediction(forest, row, class_labels)
        actual = row[class_attr]
        confusion_matrix.at[actual, predicted] += 1

    return confusion_matrix

if __name__ == "__main__":
    args = parser.parse_args()
    df = pandas.read_csv(args.dataset, index_col=None)
    df, class_attr, class_labels, attributes, attr_domain_dict = c45.preprocess_data(df)
    # 10-fold
    data_subsets = validate.generate_data_subsets(df, 10)
    matrix_array = []
    stats_array = []
    threshold = 0.06
    gr = True
    for i in range(len(data_subsets)):
        test = data_subsets[i]
        training = validate.get_training_dataset(i, data_subsets)
        forest = generate_forest(training, class_attr, class_labels, attributes, attr_domain_dict,
                                 threshold, int(args.numdataset), int(args.numattributes), int(args.numtrees),
                                 gr=gr)
        matrix = predict_on_dataset(test, forest, class_attr, class_labels)
        stats = classify.calculate_stats(matrix, test)
        matrix_array.append(matrix)
        stats_array.append(stats)
    r_matrix, r_stats, acc, err, avg_accuracy = validate.combine_matrix_and_stats(matrix_array, stats_array, data_subsets)
    print('\nRESULTS')
    print(r_matrix)
    print(f"Overall Accuracy: {round(acc, 4)}")
    print(f"Overall Error Rate: {round(err, 4)}")
    print(f"Average Accuracy {avg_accuracy}")
