import Classifier as c
import InduceC45 as c45
import sys
import pandas
import statistics as st

res_file_name = 'decision_tree.json'

def generate_data_subsets(df, k):
    data_subsets = []
    data_len = df.shape[0]
    for i in range(k-1):
        if df.shape[0] < int(data_len)/k:
            subset = df.sample(n=df.shape[0])
        else:
            subset = df.sample(n=int(data_len/k))
        data_subsets.append(subset)
        df = df.drop(subset.index)
    data_subsets.append(df)
    return data_subsets

def get_training_dataset(i, data_subsets):
    df = pandas.DataFrame()
    for j  in range(len(data_subsets)):
        if j != i:
            df = df.append(data_subsets[j])
    return df

def kfold(data_subsets, attributes, threshold, class_attr, 
          class_labels, attr_domain_dict):

    matrix_array = []
    stats_array = []
    for i in range(len(data_subsets)):
        test = data_subsets[i]
        training = get_training_dataset(i, data_subsets)
        # produces json
        c45.c45_produce_json(training, attributes, threshold, class_attr,
                            class_labels, attr_domain_dict, res_file_name)

        # classify and generate statistics
        matrix, stats = c.classifier(test, res_file_name, use_column=class_attr, 
                                     outcomes=class_labels)
        matrix_array.append(matrix)
        stats_array.append(stats)
    
    base_matrix = matrix_array[0]
    base_stats = stats_array[0]
    accuracies = []
    for i in range(len(stats_array)):
        accuracies.append(stats_array[i]['accuracy'])
    avg_accuracy = st.mean(accuracies)
    for i in range(1, len(matrix_array)):
        for j in range(len(matrix_array[i])):
            for k in range(len(matrix_array[i])):
                base_matrix.iat[j, k] += matrix_array[i].iat[j, k]
        base_stats['total_correct'] += stats_array[i]['total_correct']
        base_stats['total_incorrect'] += stats_array[i]['total_incorrect']

    overall_acc = (base_stats['total_correct']/sum(len(ds) for ds in data_subsets))* 100
    overall_err = (base_stats['total_incorrect']/sum(len(ds) for ds in data_subsets))* 100
    print('\nRESULTS')
    print(base_matrix)
    print(f"Overall Accuracy: {round(overall_acc, 4)}")
    print(f"Overall Error Rate: {round(overall_err, 4)}")
    print(f"Average Accuracy {avg_accuracy}")


def all_but_one(df, attributes, threshold, class_attr, 
          class_labels, attr_domain_dict): 

    matrix_array = []
    stats_array = []
    for i in range(len(df)):
        test = df.loc[[i+2]]
        print(test)
        train = df.drop(labels=i+2, axis=0)

        # produces json
        c45.c45_produce_json(train, attributes, threshold, class_attr,
                            class_labels, attr_domain_dict, res_file_name)

        # classify and generate statistics
        matrix, stats = c.classifier(test, res_file_name, use_column=class_attr, 
                                     outcomes=class_labels)
        matrix_array.append(matrix)
        stats_array.append(stats)
    
    base_matrix = matrix_array[0]
    base_stats = stats_array[0]
    accuracies = []
    for i in range(len(stats_array)):
        accuracies.append(stats_array[i]['accuracy'])
    avg_accuracy = st.mean(accuracies)
    for i in range(1, len(matrix_array)):
        for j in range(len(matrix_array[i])):
            for k in range(len(matrix_array[i])):
                base_matrix.iat[j, k] += matrix_array[i].iat[j, k]
        base_stats['total_correct'] += stats_array[i]['total_correct']
        base_stats['total_incorrect'] += stats_array[i]['total_incorrect']

    overall_acc = (base_stats['total_correct']/len(df))* 100
    overall_err = (base_stats['total_incorrect']/len(df))* 100
    print('\nRESULTS')
    print(base_matrix)
    print(f"Overall Accuracy: {round(overall_acc, 4)}")
    print(f"Overall Error Rate: {round(overall_err, 4)}")
    print(f"Average Accuracy {avg_accuracy}")

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
    threshold = 0.2
    
    if k > 1:
        data_subsets = generate_data_subsets(df, k)
        kfold(data_subsets, attributes, threshold, class_attr, class_labels, attr_domain_dict)
    elif k == -1:
        all_but_one(df, attributes, threshold, class_attr, 
          class_labels, attr_domain_dict)
    else:
        pass

