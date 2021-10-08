Taylor Bedrosian
Ryan Hunter

File structure for intended execution:
lab2
  |
  | — apriori.py
  | — sparse_to_binary.py
  | — goods.csv
  | — authorlist.psv
  | — bingoBaskets.csv
  |
  | — — — 1000
  |     | — 1000-out2.csv 
  |
  | — — — 5000
  |     | — 5000-out2.csv
  |
  | — — — 20000
  |     | — 20000-out2.csv
  |
  | — — — 75000
  |     | — 75000-out2.csv


Input Format:
Our code runs using Python3.8 (Conda3), with the following command line arguments and potential values:
'--dataset', either ‘bingo’ for the fantasy bingo dataset or a number(‘1000’, ‘5000’, ‘20000’, 75000’) for the corresponding Extended Bakery dataset.
'--minrs', the minSup value as a decimal < 1
'--minconf', the minConf value as a decimal < 1

For example, to print the Extended Bakery 20000 dataset results, the command would look like: 
python3 apriori.py --dataset 20000 --minrs 0.035 --minconf 0.7
