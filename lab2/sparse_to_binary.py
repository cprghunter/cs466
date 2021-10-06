import pandas
import numpy
import argparse
import itertools
import copy
import matplotlib.pyplot as plt
import io

def convert_sparse_to_binarydf(file, num_items):
    '''File is a CSV, num_items is number of distinct items in file'''
    sparse_file = open(file, 'r')
    two_d = []
    for line in sparse_file: 
        row_array = [0] * (num_items + 1) # +1 to account for index column
        sparse_line = list(pandas.read_csv(io.StringIO(line)))
        convert_to_row(row_array, sparse_line)
        two_d.append(row_array)
    df = pandas.DataFrame(two_d)
    df = df.drop(0, axis=1)
    
    return df

def convert_to_row(row_array, sparse_line):
    row_array[0] = round(float(sparse_line[0].strip()))
    for i in range(1, len(sparse_line)):
        row_array[round(float(sparse_line[i].strip()))] = 1

if __name__ == '__main__':
    fname = 'bingoBaskets.csv'
    convert_sparse_to_binarydf(fname, 1411)