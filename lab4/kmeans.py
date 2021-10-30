import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from dbscan import plot_clusters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", required=True)
    parser.add_argument("-k", "--k", required=True, type=int, help="number of clusters")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-t", "--threshold", required=True, type=float, help="threshold for stopping")
    
    args = parser.parse_args()

    return args

def centroid(df):
    return df.mean(axis=0)

def select_initial_centroids(df, k):
    s = random.randint(int(round(2*k)), int(round(3*k)))
    sample = df.sample(s)
    initial_centroid = centroid(sample)
    furthest  = np.argmax(np.linalg.norm(np.add(-df, np.array(initial_centroid))), axis=0)
    centroid_df = pd.DataFrame([df.loc[furthest]])
    centroids = {furthest}
    
    for i in range(k-1):
        furthest_pt = max([(idx, np.sum(np.linalg.norm(np.add(-np.array(row), centroid_df), axis=1)))
                        for (idx, row) in df[~df.index.isin(centroids)].iterrows()], key=lambda kv: kv[1])[0]
        centroids.add(furthest_pt)
        centroid_df.at[i+1, :] = furthest_pt
    
    return (centroids, centroid_df)
    
def check_stopping_conditions(centroids_old, centroids_new, threshold):
    # [c1, c2, c3] [c1', c2', c3']

    sum_dist = np.sum(np.linalg.norm(centroids_old-centroids_new, axis=1))
    print("sum dist: {}".format(sum_dist))

    return sum_dist <= threshold
    

def recompute_centroids(df, centroid_df, k):
    labels = pd.Series([np.argmin(np.linalg.norm(np.add(-np.array(row), centroid_df), axis=1))
                        for (_, row) in df.iterrows()], index=df.index)

    centroid_df_rows = [None]*k
    for (i, c) in enumerate(labels.unique()):
        centroid_df_rows[i] = centroid(df[labels == c])
    
    centroid_df_new = pd.DataFrame(centroid_df_rows) 

    return (centroid_df_new, labels)

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.datafile)

    centroids, centroid_df = select_initial_centroids(df, args.k)
    centroid_df_new, lbls = recompute_centroids(df, centroid_df, args.k)

    n = 0
    while not check_stopping_conditions(centroid_df, centroid_df_new, args.threshold):
        print("iteration: {}".format(n))
        print(centroid_df)
        print(centroid_df_new)
        centroid_df = centroid_df_new
        centroid_df_new, lbls = recompute_centroids(df, centroid_df_new, args.k)
        n += 1
    
    print(centroid_df_new)
    print(lbls)

    if args.plot:
        plot_clusters(df, lbls)