import Classifier as c
import sys

def all_but_one(training_set):
    return 0

if __name__ == "__main__":
    if sys.argc == 4:
        training_file = sys.argv[1]
        restr_file = sys.argv[2]
        k = int(sys.argv[3])
    else:
        training_file = sys.argv[1]
        k = int(sys.argv[3])
    if k == -1:
        all_but_one(training_file)
