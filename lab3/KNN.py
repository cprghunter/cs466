import pandas as pd
import numpy as np
import sys
import math
import validation as validate
from collections import Counter
import statistics as st

def knn(dataset, k, test_data, use_column, outcomes, normalize = False):

    if normalize == True:
        dataset = normalize_frame(dataset)
        test_data = normalize_frame(test_data)

    confusion_matrix = pd.DataFrame(0, index=outcomes, columns=outcomes)
    confusion_matrix.columns.name = "Actual \\ Classified"
    result_matrix = fill_matrix(confusion_matrix, k, dataset, test_data, str(use_column))
    print(result_matrix)
    stats = calculate_stats(result_matrix, test_data)
    print(stats)
    return result_matrix, stats

def normalize_frame(df):
    result = df
    for feature_name in df.columns:
        try:
            max_value = pd.to_numeric(df[feature_name]).max()
            min_value = pd.to_numeric(df[feature_name]).min()
            result[feature_name] = (pd.to_numeric(df[feature_name])-min_value)/(max_value-min_value)
        except:
            pass
    return result

def calculate_stats(matrix, df):
    total_classified = len(df)
    print(f"Total Classified: {total_classified}")
    total_correct = 0
    for i in range(0, len(matrix.columns)):
        total_correct += matrix.iat[i,i]
    
    stats = {}

    print(f"Total Correctly Classified: {total_correct}")
    stats['total_correct'] = total_correct
    print(f"Total Incorrectly Classified {total_classified-total_correct}")
    stats['total_incorrect'] = (total_classified-total_correct)
    accuracy = (total_correct/(total_classified)) * 100
    err_rate = (1-(total_correct/total_classified)) * 100
    stats['accuracy']=(round(accuracy, 3))
    stats['err_rate']=(round(err_rate, 3))
    print(f"Accuracy: {round(accuracy, 3)}%, Error Rate: {round(err_rate, 3)}%")
    return stats

def fill_matrix(matrix, k, dataset, test_data, use_column):
    for index, row in test_data.iterrows():
        actual = row[use_column]
        predicted = predict_k(dataset, k, row, use_column)
        matrix.at[actual, predicted] += 1
    return matrix

def predict_k(dataset, k, target_row, use_column):
    distances = []

    to_sum=[]
    to_dice=[]
    for i in target_row.keys():
        if i != use_column:
            try:
                to_sum.append(np.square(pd.to_numeric(dataset[i])-float(target_row.at[i])))
            except:
                try:
                    to_dice.append(dataset[i].str.count(target_row.at[i]))
                except:
                    pass

    new_df = pd.concat(to_sum, axis=1)    
    if to_dice:
        binary_concat = pd.concat(to_dice, axis=1)
        dice_concat = 1-(np.sum(binary_concat, axis=1)/len(binary_concat.columns))
        new_df = pd.concat([new_df, dice_concat], axis=1)
    distances.append(np.sqrt(np.sum(new_df, axis=1)))
    """
    for index, entry in dataset.iterrows():
        to_sum =[]    
        matched_strings = 0
        total_strings = 0
        for i in target_row.keys():
            if i != use_column:
                try: 
                    to_sum.append(abs(float(entry.at[i]) - float(target_row.at[i])) ** 2)
                except:
                    if entry.at[i] == target_row[i]:
                        matched_strings += 1
                    total_strings += 1
        if total_strings > 0:
            dice = matched_strings/total_strings
            to_sum.append(dice)
        distances.append(math.sqrt(sum(to_sum)))
    """
    #Now find most frequent result
    argray = np.argsort(distances).tolist()
    k_neighbors = []
    for i in argray[0][0:k]:
        k_neighbors.append(dataset.iloc[i][use_column])
    c = Counter(k_neighbors)
    return(list(dict(c).keys())[0])
    
def knn_validate(data_subsets, k, class_attr, class_labels, normalize):
    matrix_array = []
    stats_array = []
    for i in range(len(data_subsets)):
        test = data_subsets[i]
        training = validate.get_training_dataset(i, data_subsets)

        # classify and generate statistics
        matrix, stats = knn(training, k, test, class_attr, class_labels, normalize)
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

if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    k = int(sys.argv[2])
    normalize = bool(int(sys.argv[3]))
    class_attr = df.iloc[1][0]
    df = df.drop(labels=[0, 1])
    class_labels = df[class_attr].unique()
    df.dropna(inplace=True)    
    print(df)
    test_data = validate.generate_data_subsets(df, 10) #10-fold validation
    knn_validate(test_data, k, class_attr, class_labels, normalize)