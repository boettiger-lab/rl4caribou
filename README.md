# Caribou RL

A DRL-based approach to Caribou conservation based on the methods of the 
[approx-model-or-approx-soln](https://github.com/boettiger-lab/approx-model-or-approx-soln) project:

[![DOI](https://zenodo.org/badge/572256056.svg)](https://zenodo.org/badge/latestdoi/572256056) 

A three-species foodweb is considered, including interactions between Caribou, Elk and Wolf populations.

## Installation

```
git clone https://github.com/boettiger-lab/rl4caribou.git
cd rl4caribou
pip install .
```
## Train an agent

```
python scripts/train.py -f path/to/config/file.yml -id "string saving id" [-pb if you want a progress bar displayed]
```

An quick example:

```
python scripts/train.py -f hyperpars/example.yml -id "my_first_agent" -pb
```
