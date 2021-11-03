import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
#from scipy.spatial import distance_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", required=True)
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-t", "--threshold", required=True, type=float, help="threshold for stopping")
    
    args = parser.parse_args()

    return args

def distance_matrix(df1, df2):
    d_matrix_rows = []
    for idx, row in df1.iterrows():
        d_matrix_rows.append(np.linalg.norm(np.add(df2, -row).values, axis=1))
    return np.array(d_matrix_rows)

def calculate_centroid(cluster, df):
    clusterdf = df[df.index.isin(cluster)]
    return clusterdf.mean(axis=0)

def combine_clusters(to_combine, clusters, clusters_by_dist, centroid_df, max_idx, df, dist_df):
    new_cluster = clusters[to_combine[0]].union(clusters[to_combine[1]])
    distance = dist_df.loc[to_combine]
    centroid_df.drop(list(to_combine), axis=0, inplace=True)
    max_idx += 1
    clusters[max_idx] = new_cluster
    del clusters[to_combine[0]]
    del clusters[to_combine[1]]
    clusters_by_dist[distance] = frozenset(clusters.values())

    centroid = calculate_centroid(new_cluster, df)
    centroid_df.at[max_idx] = centroid
    return max_idx, centroid_df

def get_cluster_labels(df, clusters):
    cluster_labels = [None for i,_ in df.iterrows()]
    for i, fset in enumerate(clusters): 
        for idx in fset:
            cluster_labels[idx] = i
    return cluster_labels

def plot_clusters(df, cluster_labels):
    plt.scatter(df[df.columns[0]], df[df.columns[1]], c=cluster_labels)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.datafile)
    max_idx = len(df)

    clusters = {idx: frozenset({idx}) for idx, _ in df.iterrows()}
    clusters_by_dist = {}
    clusters_by_dist[0] = frozenset(clusters.values())

    dist_df = pd.DataFrame(distance_matrix(df, df), index=df.index, columns=df.index)
    mindist = dist_df.where(dist_df != 0).stack().idxmin()
    centroid_df = df.copy()
    max_idx, centroid_df = combine_clusters(mindist, clusters, clusters_by_dist, centroid_df, max_idx, df, dist_df)
    while len(centroid_df) > 1:
        d_matrix = distance_matrix(centroid_df, centroid_df)
        dist_df = pd.DataFrame(d_matrix, 
                                    index=centroid_df.index, columns=centroid_df.index)
        mindist = dist_df.where(dist_df != 0).stack().idxmin() 
        max_idx, centroid_df = combine_clusters(mindist, clusters, clusters_by_dist, centroid_df, max_idx, df, dist_df)

    if args.threshold in clusters_by_dist:
        plot_clusters(df, get_cluster_labels(df, clusters_by_dist[args.threshold]))
        print(clusters_by_dist[args.threshold])
    else:
        sorted_keys = list(sorted(clusters_by_dist.keys()))
        for i, key in enumerate(sorted_keys):
            if key > args.threshold:
                plot_clusters(df, get_cluster_labels(df, clusters_by_dist[sorted_keys[i - 1]]))
                print(clusters_by_dist[sorted_keys[i - 1]])
                break
