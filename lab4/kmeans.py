import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import argparse
import random
from dbscan import plot_clusters
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datafile", required=True)
    parser.add_argument("-k", "--k", required=True, type=int, help="number of clusters")
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-p3d", "--plot3d", action="store_true")
    parser.add_argument("-c", "--class_labels", type=str, help="file containing class labels for data file")

    parser.add_argument("-t", "--threshold", required=True, type=float, help="threshold for stopping")
    parser.add_argument("-f", "--try_params", action="store_true")
    parser.add_argument("-m", "--max_k", type=int)
    parser.add_argument("-purity", "--purity", action="store_true")
    
    args = parser.parse_args()

    return args

def centroid(df):
    return df.mean(axis=0)

def select_initial_centroids(df, k):
    s = random.randint(min(len(df), int(round(2*k))), min(len(df), int(round(3*k))))
    sample = df.sample(s)
    initial_centroid = centroid(sample)

    furthest = np.argmax(np.linalg.norm(np.add(-np.array(initial_centroid), sample), axis=1))
    centroid_df = pd.DataFrame(np.zeros((k, len(df.columns))), index=list(range(k)), columns=df.columns)
    centroid_df.loc[0] = sample.iloc[furthest]
    centroids = {furthest}
    
    for i in range(0, k-1):
        sum_dists = []
        for (idx, row) in sample[~sample.index.isin(centroids)].iterrows():
            sum_dist = np.sum(np.linalg.norm(np.add(-np.array(row), centroid_df), axis=1))
            sum_dists.append((idx, sum_dist))

        m = max(sum_dists, key=lambda kv: kv[1])
        furthest_pt = m[0]
        centroids.add(furthest_pt)
        centroid_df.loc[i+1, :] = sample.loc[furthest_pt]
    return (centroids, centroid_df)
    
def check_stopping_conditions(centroids_old, centroids_new, threshold):
    sum_dist = np.sum(np.linalg.norm(centroids_old-centroids_new, axis=1))

    return sum_dist <= threshold

def recompute_centroids(df, centroid_df, k):
    labels = pd.Series([np.argmin(np.linalg.norm(np.add(-np.array(row), centroid_df), axis=1))
                        for (_, row) in df.iterrows()], index=df.index)

    #centroid_df_rows = [None]*k
    centroid_df_new = pd.DataFrame(np.zeros((k, len(df.columns))), index=centroid_df.index, columns=df.columns) 
    uniq = labels.unique()
    for idx in centroid_df.index:
        if idx not in uniq:
            centroid_df_new.loc[idx] =  centroid_df.loc[idx]
        else:
            centroid_df_new.loc[idx] = centroid(df[labels == idx])

    return (centroid_df_new, labels)

def kmeans(df, k, threshold):
    _, centroid_df = select_initial_centroids(df, k)
    centroid_df_new, lbls = recompute_centroids(df, centroid_df, k)
    n = 0
    while not check_stopping_conditions(centroid_df, centroid_df_new, threshold):
        centroid_df = centroid_df_new
        centroid_df_new, lbls = recompute_centroids(df, centroid_df_new, k)
        n += 1
    
    return lbls

def create_k_sse_plot(df, max_k, threshold):
    x = []
    y = []
    for k in range(1, max_k):
        x.append(k)
        lbls = kmeans(df, k, threshold)
        sse = utils.total_sse(df, utils.collect_clusters(df, lbls))
        y.append(sse)
    plt.xlabel('k')
    plt.ylabel('Total SSE') 
    plt.xticks(np.arange(1, k+1))
    plt.plot(x, y)
    plt.show()

def plot_clusters_3d(df, lbls):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df[df.columns[0]], df[df.columns[1]], df[df.columns[2]], c=lbls)
    plt.show()

def main():
    args = parse_args()
    data_df, labels_df = utils.parse_csv(args.datafile)

    if args.try_params:
        create_k_sse_plot(data_df, args.max_k, args.threshold)

    lbls = kmeans(data_df, args.k, args.threshold)

    if args.plot:
        plot_clusters(data_df, lbls)
    elif args.plot3d:
        plot_clusters_3d(data_df, lbls)
    
    utils.print_cluster_stats(data_df, utils.collect_clusters(data_df, lbls))
    if args.purity:
        utils.report_cluster_purity(data_df, labels_df, lbls)

if __name__ == "__main__":
    main()