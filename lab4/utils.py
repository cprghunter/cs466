import pandas as pd
import numpy as np
from collections import defaultdict

def collect_clusters(df, labels) -> dict[int, pd.Series]:
    clusters = defaultdict(lambda: set())
    for (i, lbl) in enumerate(labels):
        clusters[lbl].add(df.index[i])
    
    return clusters

def calculate_centroid(cluster, df):
    clusterdf = df[df.index.isin(cluster)]
    return clusterdf.mean(axis=0)

def print_cluster_stats(df, clusters, key_col=None):
    sse_tot = 0
    for (lbl, cluster) in sorted(clusters.items()):
        print(f"Cluster: {lbl}: ")
        centroid = calculate_centroid(cluster, df)
        print(f"Center: {format_row(centroid)}")
        mx, mn, avg = max_min_avg_dist(df, cluster, centroid)
        print(f"distances to center (min, max, average): ({mn}, {mx}, {avg})")
        print(f"{len(cluster)} points:")
        print_cluster(df, cluster, key_col)

        sse = calculate_sse(df, cluster, centroid)
        sse_tot += sse
    
    #print(f"Silhouette Score: {silhouette(df, clusters)}")
    print(f"SSE: {sse_tot}")

def format_row(row):
    return ",".join(map(str, row.values))

def print_cluster(df, cluster, key_col=None, taboff=1):
    for row_idx in sorted(cluster):
        row = df.loc[row_idx]
        print("\t"*taboff + f"{row[key_col] if key_col else format_row(row)}")

def calculate_sse(df, cluster, centroid) -> float:
    cluster_pts = df[df.index.isin(cluster)] 
    return np.sum(np.sum(np.square(np.add(-np.array(centroid), cluster_pts))))

def max_min_avg_dist(df, cluster, centroid) -> float: 
    cluster_pts = df[df.index.isin(cluster)] 
    dists = np.linalg.norm(np.add(-np.array(centroid), cluster_pts), axis=1)
    return np.max(dists), np.min(dists), np.mean(dists)

def parse_line(line, reader, is_numeric):
    parts = line.split(",")
    numeric = [reader(part.strip()) for i, part in enumerate(parts) if is_numeric[i]]
    non_numeric = [str(part.strip()) for i, part in enumerate(parts) if not is_numeric[i]]
        
    return (numeric, non_numeric)

def parse_csv(file_name):
    with open(file_name, 'r') as f:
        contents = f.read()
    lines = contents.split('\n')
    header = list(map(lambda part: int(part.strip()), lines[0].split(",")))
    print(header)
    rows = []
    index = []
    for line in map(lambda line: line.strip(), lines[1:]):
        if not line:
            continue
        numeric, non_numeric = parse_line(line, np.float64, header)
        if len(non_numeric) == 1:
            index.append(non_numeric[0])
        rows.append(numeric)
    columns = list(range(len(rows[0]))) 
    if len(index) == 0:
        index = list(range(len(rows)))
    return pd.DataFrame(rows, index=index, columns=columns)

def distance_matrix(df1, df2):
    d_matrix_rows = []
    for idx, row in df1.iterrows():
        d_matrix_rows.append(np.linalg.norm(np.add(df2, -row).values, axis=1))
    return np.array(d_matrix_rows)

def total_sse(df, clusters):
    return np.sum([calculate_sse(df, cluster, calculate_centroid(cluster, df)) for _, cluster in clusters.items()])

def silhouette(df, clusters):
    centroids = [calculate_centroid(cluster, df) for lbl, cluster in sorted(clusters.items())]
    centroid_df = pd.DataFrame(centroids, index=sorted(clusters.keys()))
    dist_df = pd.DataFrame(distance_matrix(centroid_df, centroid_df), 
        centroid_df.index, columns=centroid_df.index)
    
    a = 0
    b = 0
    for lbl, centroid in centroid_df.iterrows():
        cluster_pts = df[df.index.isin(clusters[lbl])]
        mean_intra_dist = np.mean(np.linalg.norm(np.add(-np.array(centroid), cluster_pts), axis=1))
        a += mean_intra_dist
        min_dist_centroid = centroid_df.loc[dist_df.loc[lbl].where(dist_df.loc[lbl].index != lbl).idxmin()]
        mean_nearest_dist = np.mean(np.linalg.norm(np.add(-np.array(min_dist_centroid), cluster_pts), axis=1))
        b += mean_nearest_dist
    
    print(a)
    print(b)
    
    return (b-a) / max(a, b)