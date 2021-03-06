import pandas as pd
import numpy as np
import sys
import json

def classifier(dataset, tree=None, use_column=None, outcomes=None):
    if type(dataset) == str:
        df = pd.read_csv(dataset)
        if not pd.isna(df.iloc[1,0]):
            use_column = df.iloc[1,0]
            df = df.drop(labels=[0,1])
            outcomes = df.loc[:, str(use_column)].unique()
            make_matrix = True
        else:
            df.drop(labels=[0, 1])
            make_matrix = False
    else:
        df = dataset
        make_matrix = True
    if type(tree) == str: 
        open_tree = open(tree, 'r')
        decision_tree = json.load(open_tree)
    elif type(tree) == dict:
        decision_tree = tree
    
    if make_matrix:
        confusion_matrix = pd.DataFrame(0, index=outcomes, columns=outcomes)
        confusion_matrix.columns.name = "Actual \\ Classified"
        result_matrix = fill_matrix(confusion_matrix, decision_tree, df, str(use_column))
        print(result_matrix)
        stats = calculate_stats(result_matrix, df)
        print(stats)
        return result_matrix, stats
    else:
        just_predict(decision_tree, df)

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


def just_predict(tree, data): #TODO
    for index, row in data.iterrows():
        print(row.to_string())
        print(f"Predicted: {traverse_tree(tree, row)}\n")

def fill_matrix(matrix, tree, data, use_column):
    for index, row in data.iterrows():
        actual = row[use_column]
        predicted = traverse_tree(tree, row)
        matrix.at[actual, predicted] += 1
    return matrix

def traverse_tree(tree, row):
    if 'node' in tree.keys(): 
        find_edge = row[tree['node']['var']]
        for edge in tree['node']['edges']:
            if 'direction' in edge['edge'].keys():
                if (edge['edge']['direction'] == 'le') and (find_edge <= edge['edge']['value']):
                    return traverse_tree(edge['edge'], row)
                elif (edge['edge']['direction'] == 'gt') and (find_edge >= edge['edge']['value']):
                    return traverse_tree(edge['edge'], row)
            else:
                if edge['edge']['value'] == find_edge:
                    return traverse_tree(edge['edge'], row)
    else:
        return tree['leaf']['decision']


if __name__ == '__main__':
    classifier(dataset = sys.argv[1], tree = sys.argv[2])
