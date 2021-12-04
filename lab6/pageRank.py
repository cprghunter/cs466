from collections import defaultdict
import pandas as pd
import numpy as np
import argparse
import datetime
from scipy.sparse import csr_matrix


def get_node_name_to_idx(node_array):
    node_to_idx = {}
    for i, node in enumerate(node_array):
        node_to_idx[node] = i
    return node_to_idx

def get_degrees(df, node_array, directed=False):
    node1_vc = df["node1"].value_counts()
    node2_vc = df["node1"].value_counts()
    node_degrees = defaultdict(lambda x : 1)
    for node in node_array:
        if not directed:
            try:
                node_degrees[node] = node1_vc[node] + node2_vc[node]
            except KeyError:
                try:
                    node_degrees[node] = node1_vc[node]
                except KeyError:
                    node_degrees[node] = node2_vc[node]

        else:
            try:
                node_degrees[node] = node1_vc[node]
            except KeyError:
                pass
    return node_degrees

def get_adj_unweighted_undirected(df, node_to_idx, node_degree):
    row = []
    col = []
    data = []
    for idx, edge in df.iterrows():
        row.append(node_to_idx[edge["node1"]]) 
        col.append(node_to_idx[edge["node2"]])
        data.append(1/node_degree[edge["node2"]])

        row.append(node_to_idx[edge["node2"]]) 
        col.append(node_to_idx[edge["node1"]])
        data.append(1/node_degree[edge["node1"]])
    return row, col, data

def get_adj_weighted_undirected(df, node_to_idx, node_degree):
    row = []
    col = []
    data = []
    for idx, edge in df.iterrows():
        if edge["node1_value"] != 0:
            row.append(node_to_idx[edge["node1"]]) 
            col.append(node_to_idx[edge["node2"]])
            data.append(edge["node1_value"])

            row.append(node_to_idx[edge["node2"]]) 
            col.append(node_to_idx[edge["node1"]])
            data.append(edge["node1_value"])
        else:
            row.append(node_to_idx[edge["node1"]]) 
            col.append(node_to_idx[edge["node2"]])
            data.append(edge["node2_value"])

            row.append(node_to_idx[edge["node2"]]) 
            col.append(node_to_idx[edge["node1"]])
            data.append(edge["node2_value"])
    return row, col, data

def get_adj_weighted_directed(df, node_to_idx, node_degree):
    row = []
    col = []
    data = []
    for idx, edge in df.iterrows():
        if edge["node1_value"] > edge["node2_value"]:
            row.append(node_to_idx[edge["node2"]]) 
            col.append(node_to_idx[edge["node1"]])
            data.append(edge["node1_value"] / (edge["node1_value"] + edge["node2_value"]))
        else:
            row.append(node_to_idx[edge["node1"]]) 
            col.append(node_to_idx[edge["node2"]])
            data.append(edge["node2_value"] / (edge["node1_value"] + edge["node2_value"]))
    return row, col, data

def create_empty_adj_matrix(df):
    nodes = np.union1d(df["node1"].unique(), df["node2"].unique())
    adj = pd.DataFrame(0, index=nodes, columns=nodes, dtype=np.float16)
    return adj

def get_adjacency_matrix(data_file):
    cols = ["node1", "node1_value", "node2", "node2_value"]
    df = pd.read_csv(data_file, names=cols, index_col=False, quoting=0, quotechar="\"", skipinitialspace=True)
    # use node values to determine what kind of data we're looking at
    node1_values = df["node1_value"].unique()
    node2_values = df["node2_value"].unique()
    nodes = np.union1d(df["node1"].unique(), df["node2"].unique())

    node_to_idx = get_node_name_to_idx(nodes)
    if len(node2_values) == 1: # no second edge weight, graph is undirected
        node_degree = get_degrees(df, nodes)
        if len(node1_values) == 1: # no first edge weights, graph is unweighted
            row, col, data = get_adj_unweighted_undirected(df, node_to_idx, node_degree)
        else:
            row, col, data = get_adj_weighted_undirected(df, node_to_idx, node_degree)
    else: # directional, weighted, and every edge has a corresponding inverse edge like NCAA
        node_degree = get_degrees(df, nodes, directed=True)
        row, col, data = get_adj_weighted_directed(df, node_to_idx, node_degree)
    adj_matrix = csr_matrix((data, (row, col)), shape=(nodes.shape[0], nodes.shape[0]), dtype=np.float64)
    return adj_matrix, nodes

def get_adjacency_matrix_snap(data_file):
    rows=[]
    cols=[]
    data=[]
    with open(data_file, 'r') as f:
        line = f.readline()
        while line and line[0] == "#":
            line = f.readline()
        while line:
            edge = line.split()
            rows.append(int(edge[0]))
            cols.append(int(edge[1]))
            data.append(1)
            line = f.readline()
    n = max(max(rows), max(cols))
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n+1,n+1), dtype=np.float64)
    return adj_matrix, list(range(n+1))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help="csv file to run version of pageRank on")
    args = parser.parse_args()
    read_time_start = datetime.datetime.now()
    if args.data[-3:] == "txt":
        snap = True
        adj_matrix, nodes = get_adjacency_matrix_snap(args.data)
    else:
        snap = False
        adj_matrix, nodes = get_adjacency_matrix(args.data)
    read_time_total = datetime.datetime.now() - read_time_start

    process_time_start = datetime.datetime.now() 
    sums = adj_matrix.sum(axis=1)
    #other_sums = [adj_matrix[i].sum() for i in range(adj_matrix.shape[0])] 
    #other_inverse = [1/other_sums[i] if other_sums[i] != 0 else 0 for i in range(len(other_sums))]
    array_sums_inverse = np.array([1/sums[i,0] if not sums[i,0] == 0 else 0 for i in range(sums.shape[0])])
    """
    for i in range(adj_matrix.shape[0]):
        adj_matrix[i] = adj_matrix[i] / adj_matrix[i].sum()
    """
    adj_matrix = adj_matrix.transpose()

    adj_matrix = adj_matrix.multiply(array_sums_inverse) 

    iterations = 1
    d = 0.95
    page_ranks_old = np.array([1/adj_matrix.shape[0] for _ in range(adj_matrix.shape[0])])
    page_rank = adj_matrix.dot(page_ranks_old)* d + (1 - d) * (1 / page_ranks_old.shape[0])
    while np.sum(np.absolute(page_ranks_old - page_rank)) >= 0.000001:
        page_ranks_old = page_rank
        page_rank = adj_matrix.dot(page_ranks_old) * d + (1 - d) * (1 / page_ranks_old.shape[0])
        iterations += 1 
    ranks = {nodes[i]: page_rank[i] for i in range(page_rank.shape[0])}
    sorted = sorted(ranks, key=ranks.get, reverse=True)
    process_time_total = datetime.datetime.now() - process_time_start
    if snap:
        with open("ranks-out.txt", 'w') as f:
            for node in sorted:
                f.write(f"{node} with {ranks[node]:.14f}\n")
    else: 
        for node in sorted:
            print(f"{node} with {ranks[node]:.4f}")
    print(page_rank.shape[0])
    print(f"iterations: {iterations}")
    print(f"read time: {read_time_total}")
    print(f"process time: {process_time_total}")
