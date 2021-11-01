import pandas as pd
import numpy as np
import argparse
from scipy.spatial import distance_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", required=True)
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-t", "--threshold", required=True, type=float, help="threshold for stopping")
    
    args = parser.parse_args()

    return args

def calculate_centroid(cluster, df):
    clusterdf = df[df.index.isin(cluster)]
    return clusterdf.mean(axis=0)

def combine_clusters(to_combine, clusters, centroid_df, max_idx, df):
    new_cluster = clusters[to_combine[0]].union(clusters[to_combine[1]])
    centroid_df.drop(list(to_combine), axis=0, inplace=True)
    max_idx += 1
    clusters[max_idx] = new_cluster
    centroid = calculate_centroid(new_cluster, df)
    centroid_df.at[max_idx] = centroid
    return max_idx, centroid_df

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.datafile)
    max_idx = len(df)

    clusters = {idx: {idx} for idx, _ in df.iterrows()}

    dist_df = pd.DataFrame(distance_matrix(df, df), index=df.index, columns=df.index)
    mindist = dist_df.where(dist_df != 0).stack().idxmin()
    centroid_df = df.copy()
    max_idx, centroid_df = combine_clusters(mindist, clusters, centroid_df, max_idx, df)
    while len(centroid_df) > 1:
        d_matrix = distance_matrix(centroid_df, centroid_df)
        dist_df = pd.DataFrame(d_matrix, 
                                    index=centroid_df.index, columns=centroid_df.index)
        mindist = dist_df.where(dist_df != 0).stack().idxmin()
        max_idx, centroid_df = combine_clusters(mindist, clusters, centroid_df, max_idx, df)
        #print(f"index: {centroid_df.index}")
    print(clusters[max_idx])
    print(centroid_df)
