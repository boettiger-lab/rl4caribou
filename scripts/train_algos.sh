#!/bin/bash

# move to script directory for normalized relative paths.
scriptdir="$(dirname "$0")"
cd "$scriptdir"

python train.py -f ../hyperpars/ppo-caribou.yml &
python train.py -f ../hyperpars/rppo-caribou.yml &
python train.py -f ../hyperpars/tqc-caribou.yml &
python train.py -f ../hyperpars/td3-caribou.yml &
