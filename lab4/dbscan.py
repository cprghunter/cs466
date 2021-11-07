import pandas as pd
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import utils

def density_connected(point_idx, e_neighborhoods, cores, cluster_labels, label):
    for neighbor_idx in e_neighborhoods[point_idx]:
        if cluster_labels[neighbor_idx] < 0:
            cluster_labels[neighbor_idx] = label
            if neighbor_idx in cores:
                density_connected(neighbor_idx, e_neighborhoods, cores, 
                                    cluster_labels, label)
 
def calculate_e_neighborhood(datapoint, df, e):
    dists = np.linalg.norm(np.add(-np.array(datapoint),df), axis=1)
    return np.where(dists <= e)[0]

def plot_clusters(df, cluster_labels):
    cluster_labels = list(map(lambda lbl: None if lbl < 0 else lbl, cluster_labels))
    plt.scatter(df[df.columns[0]], df[df.columns[1]], c=cluster_labels)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", required=True)
    parser.add_argument("-e", "--epsilon", required=True, type=float)
    parser.add_argument("-min_e", "--min_e", type=float)
    parser.add_argument("-max_e", "--max_e", type=float)
    parser.add_argument("-n", "--numpoints", required=True, type=int)
    parser.add_argument("-max_n", "--max_n", type=int)
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-purity", "--purity", action="store_true")
    args = parser.parse_args()

    return args

def dbscan(df_data, epsilon, numpoints):
    e_neighborhoods = {}
    cluster_labels = []
    for idx, row in df_data.iterrows():
        e_neighborhoods[idx] = calculate_e_neighborhood(row, df_data, epsilon)
        e_neighborhoods[idx] = e_neighborhoods[idx][e_neighborhoods[idx] != idx]
        cluster_labels.append(-1)
    cores = set([idx for idx in e_neighborhoods 
                     if len(e_neighborhoods[idx]) >= numpoints])
    current_cluster = 0
    print(cluster_labels)
    for core_idx in cores:
        if cluster_labels[core_idx] < 0:
            cluster_labels[core_idx] = current_cluster
            density_connected(core_idx, e_neighborhoods, cores, 
                                                        cluster_labels, current_cluster)
            current_cluster += 1

    return cluster_labels 

def main():
    args = parse_args()
    df_data, df_labels = utils.parse_csv(args.datafile)

    lbls = dbscan(df_data, args.epsilon, args.numpoints)

    if args.plot:
        plot_clusters(df_data, lbls)
    
    utils.print_cluster_stats(df_data, utils.collect_clusters(df_data, lbls))
    if args.purity:
        utils.report_cluster_purity(df_data, df_labels, lbls)

if __name__ == "__main__":
    main()
    