import pandas
import numpy
import argparse
import itertools
import copy
import matplotlib.pyplot as plt
import sparse_to_binary as stb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--minrs', type=float)
parser.add_argument('--minconf', type=float)
args = parser.parse_args()

GOODS_FILE = "goods.csv"
BAKERY_OUT2 = "-out2.csv"
BINGO = "bingoBaskets.csv"
AUTHORS = 'authorlist.psv'


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

def itemset_in_basket(transaction, itemset):
    """
    Returns a True if itemset in transaction, False otherwise
    """
    for item in itemset:
        if not transaction[item]:
            return False
    return True

def apriori_freq(transactions, items, minsup):
    """
    transactions - df from -out2 file
    items - list from 1-50 representing bakery items and indeces of df vector
    minsup - minsupport
    """
    freq_sets = {}
    item_counts = {i: sum(transactions[i]) for i in transactions.columns}
    item_rsup = {i: item_counts[i] / len(transactions.index)
                    for i in transactions.columns}
    k = 1
    freq_sets = {'f'+str(k): round_1_candidates(items, item_rsup, minsup)}

    # start counts cache to make association rule calculations faster
    item_counts_with_set_keys = {frozenset([i]): item_counts[i] for i in transactions.columns}
    set_counts = {'f'+str(k): item_counts_with_set_keys} 
    # second dict with no skyline pruning
    all_freq_sets = copy.deepcopy(freq_sets)     
    """
    freq_sets = {f3: {frozenset(i, j, k): rsup, frozenset(x, y, z): rsup}} 
    """
    k = 2
    print(f"f1 = {len(freq_sets['f'+str(k-1)])}")
    while freq_sets.get('f'+str(k-1)):
        candidates = candidate_gen(freq_sets, k-1)
        count_dict = {candidate: 0 for candidate in candidates}
        for candidate in count_dict:
            candidate_indexes = numpy.asarray(candidate)
            for i in range(len(transactions.index)):
                if itemset_in_basket(transactions.iloc[i], candidate):
                    count_dict[candidate] += 1
        set_counts['f'+str(k)] = count_dict

        # get frequent sets from candidates
        still_frequent = {}
        rsup_all = []
        for i in count_dict:
            rsup = count_dict[i] / len(transactions.index)
            rsup_all.append(rsup)
            if rsup >= minsup:
                still_frequent[i] = rsup 
        freq_sets['f'+str(k)] = still_frequent

        if args.plot:
            plot_rsup_of_sets(rsup_all, k)

        # remove all subsets of larger sets from freq_sets to maintain skyline 
        sets_to_remove = set()
        for freq_set in still_frequent.keys():
            for smaller_freq_set in freq_sets['f'+str(k-1)].keys():
                if smaller_freq_set.issubset(freq_set):
                    sets_to_remove.add(smaller_freq_set)

        for smaller_set in sets_to_remove:
            freq_sets['f'+str(k-1)].pop(smaller_set)

        if not freq_sets['f'+str(k)]:
            print("Empty frequent set detected, ending...")
        k += 1
    return freq_sets, set_counts

def plot_rsup_of_sets(rsup_list, k):
    plt.scatter(rsup_list, [1 for _ in rsup_list])
    plt.title(f"R-Support for sets of size {k}")
    plt.show()

def calculate_conf(set_counts, combined, combined_no_right):
    return (set_counts['f'+str(len(combined))][combined] 
            / set_counts['f'+str(len(combined_no_right))][combined_no_right])

def get_assoc_rules(skyline_freq, set_counts, minconf):
    assoc_rules = {}
    for freq_set in skyline_freq:
        if len(freq_set) >= 2:
            for item in freq_set:
                item_as_set = frozenset([item])
                conf = calculate_conf(set_counts, freq_set, freq_set.difference(item_as_set))
                if conf >= minconf:
                    assoc_rules[(freq_set.difference(item_as_set), item)] = conf
    return assoc_rules

def assoc_rules_to_str(good_labels_df, assoc_rules):
    assoc_rules_str = ""
    for rule in assoc_rules:
        assoc_rules_str += (f"\n{[good_labels_df.iloc[item-1]['Flavor'] + good_labels_df.iloc[item-1]['Food'] for item in rule[0]]} ->"
                            f"{good_labels_df.iloc[rule[1] - 1]['Flavor'] + good_labels_df.iloc[rule[1] - 1]['Food']}")
        assoc_rules_str += f": {assoc_rules[rule]}"
    return assoc_rules_str
    
def bingo_rules_to_str(bingo_labels, assoc_rules):
    bingo_labels.columns = ['Name']
    for rule in assoc_rules:
        print(f"{[bingo_labels.iloc[item-2]['Name'] for item in rule[0]]} -> {bingo_labels.iloc[rule[1]-2]['Name']}")
    return

def round_1_candidates(items, item_rsup, minsup):
    candidates_and_rsup = {}
    for item in items:
        if item_rsup[item] >= minsup:
            candidate = frozenset([item])
            candidates_and_rsup[candidate] = item_rsup[item]
    return candidates_and_rsup

def consolidate_freq_set_dict(freq_sets):
    skyline_freq_sets = set()
    for fk_dict in freq_sets:
        for freq_set in freq_sets[fk_dict].keys():
            skyline_freq_sets.add(freq_set)
    return skyline_freq_sets

def candidate_gen(freq_sets, k):
    """
    freq_sets = {f3: {frozenset(i, j, k): rsup, frozenset(x, y, z): rsup}} 
    """
    c = set()
    ksets = freq_sets['f'+str(k)].keys()
    for candidate in ksets:
        for other_candidate in ksets:
            union = candidate.union(other_candidate) 
            if len(union) == k + 1:
                flag = True
                for item in union:
                    subset = union.difference(frozenset((item,)))
                    if not subset in ksets:
                        flag = False
                        break
                if flag:
                    c.add(union)
    return c

if __name__ == "__main__":
    if args.dataset == 'bingo':
        bingo_bv_df = stb.convert_sparse_to_binarydf(BINGO, 1411)
        freq_sets, set_counts = apriori_freq(bingo_bv_df, bingo_bv_df.columns, args.minrs)
        author_labels_df = pandas.read_csv('authorlist.psv', sep='|', index_col=0)
        skyline_freq = consolidate_freq_set_dict(freq_sets)
        skyline_assoc_rules = get_assoc_rules(skyline_freq, set_counts, args.minconf)
        print([set(freq) for freq in skyline_freq])
        bingo_rules_to_str(author_labels_df, skyline_assoc_rules)
    else:
        good_labels_df = load_good_labels(GOODS_FILE)
        bakery_bv_df = load_full_bv_bakery(f"{args.dataset}/{args.dataset}{BAKERY_OUT2}")
        freq_sets, set_counts = apriori_freq(bakery_bv_df, bakery_bv_df.columns, args.minrs)
        skyline_freq = consolidate_freq_set_dict(freq_sets)
        skyline_assoc_rules = get_assoc_rules(skyline_freq, set_counts, args.minconf)
        print([set(freq) for freq in skyline_freq])
        print(assoc_rules_to_str(good_labels_df, skyline_assoc_rules))
