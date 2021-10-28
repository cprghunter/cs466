import pandas
import argparse
import numpy as np

def density_connected(point_idx, e_neighborhoods, cores, cluster_labels, label):
    for neighbor_idx in e_neighborhoods[point_idx]:
        if cluster_labels[neighbor_idx] == None:
            cluster_labels[neighbor_idx] = label
            if neighbor_idx in cores:
                density_connected(neighbor_idx, e_neighborhoods, cores, 
                                    cluster_labels, label)
 
def calculate_e_neighborhood(datapoint, df, e):
    dists = np.linalg.norm(np.add(-np.array(datapoint),df), axis=1)
    return np.where(dists <= e)[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", required=True)
    parser.add_argument("-e", "--epsilon", required=True, type=float)
    parser.add_argument("-n", "--numpoints", required=True, type=int)
    args = parser.parse_args()

    df = pandas.read_csv(args.datafile)

    e_neighborhoods = {}
    cluster_labels = []
    for idx, row in df.iterrows():
        e_neighborhoods[idx] = calculate_e_neighborhood(row, df, args.epsilon)
        e_neighborhoods[idx] = e_neighborhoods[idx][e_neighborhoods[idx] != idx]
        cluster_labels.append(None)
    cores = set([idx for idx in e_neighborhoods 
                     if len(e_neighborhoods[idx]) >= args.numpoints])
    current_cluster = 0
    for core_idx in cores:
        if cluster_labels[core_idx] == None:
            cluster_labels[core_idx] = current_cluster
            density_connected(core_idx, e_neighborhoods, cores, 
                                                        cluster_labels, current_cluster)
            current_cluster += 1

    print(cluster_labels)
