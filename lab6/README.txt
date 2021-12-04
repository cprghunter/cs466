usage: pageRank.py [-h] [--data DATA]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA, -d DATA  csv or txt file to run version of pageRank on

If the file ends in .txt pageRank assumes it's a SNAP dataset and processes it as such.
No modification to epsilon or d can be done from the command line. My implementation automatically
figures out if a provided csv dataset is directed or undirected, and weighted or unweighted.