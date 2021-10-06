import pandas
import numpy
import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--file')
parser.add_argument('--minrs', type=int)
args = parser.parse_args()

GOODS_FILE = "goods.csv"
BAKERY_OUT2 = "-out2.csv"


def load_good_labels(filename):
    # labeled goods df, indexed on Id has Flavor, Food, Price, Type column
    good_labels_df = pandas.read_csv(filename, index_col="Id")
    return good_labels_df


def load_full_bv_bakery(filename):
    # indexed by transaction id, .iloc[tid][item_id] to get value
    full_bv_df = pandas.read_csv(filename, index_col=0, header=None,
                                 names=[i for i in range(51)])
    # example full_bv_df.iloc[2][2]
    return full_bv_df


def apriori_freq_first_pass(transactions, items, minsup):
    """
    transactions - df from -out2 file
    items - list from 1-50 representing bakery items and indeces of df vector
    minsup - minsupport
    """
    freq_sets = {}
    item_counts = {i: sum(transactions[i]) for i in transactions.columns}
    sum_all_items = sum(item_counts[i] for i in transactions.columns)
    item_rsup = {i: item_counts[i] /
                 sum_all_items for i in transactions.columns}
    print(item_rsup)
    k = 1
    freq_sets = {'f'+str(k): round_1_candidates(items, item_rsup, minsup)}
    k = 2
    print(freq_sets['f'+str(k-1)])
    while freq_sets.get('f'+str(k-1)):
        candidates = candidate_gen(freq_sets['f'+str(k-1)], k)
        count_dict = {candidates[i]: 0 for i in range(0, len(candidates))}
        for candidate in count_dict:
            candidate_indexes = numpy.asarray(candidate)
            for i in range(len(candidate_indexes)):
                candidate_indexes[i] -= 1
            for i in range(0, len(transactions)):
                row = transactions.iloc[i, candidate_indexes]
                if row.sum() == k:
                    count_dict[candidate] += 1

        print("DONE WITH ROUND COUNTING")
        still_frequent = []
        for i in count_dict:
            print(count_dict[i] / sum_all_items)
            if (count_dict[i] / sum_all_items) >= minsup:
                still_frequent.append(i)

        freq_sets['f'+str(k)] = still_frequent

        if freq_sets['f'+str(k)]:
            print(freq_sets['f'+str(k)], end=' ')
        else:
            print("Empty frequent set detected, ending...")

        k += 1

def round_1_candidates(items, item_rsup, minsup):
    candidates = []
    for item in items:
        if item_rsup[item] >= minsup:
            candidates.append(item)
    return candidates

def candidate_gen(candidates, k):
    '''Note, this solely does the join step, not pruning currently'''
    union = itertools.combinations(candidates, k)
    c = []
    for i in union:
        c.append(i)
    return c

if __name__ == "__main__":
    good_labels_df = load_good_labels(GOODS_FILE)
    bakery_bv_df = load_full_bv_bakery(f"1000/1000{BAKERY_OUT2}")
    apriori_freq_first_pass(bakery_bv_df, (i for i in range(1, 51)), 0.026)
