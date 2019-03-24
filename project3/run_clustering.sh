#!/bin/sh

# Replace 'X' below with the optimal values found
# If you want to first generate data and updated datasets, remove the "--skiprerun" flags below

python run_experiment.py --ica --wine --dim 6 --skiprerun --verbose --threads -1 > ica-wine-clustering.log 2>&1
python run_experiment.py --ica --diabetes --dim 12 --skiprerun --verbose --threads -1 > ica-diabetes-clustering.log 2>&1
python run_experiment.py --pca --wine --dim 3 --skiprerun --verbose --threads -1 > pca-wine-clustering.log 2>&1
python run_experiment.py --pca --diabetes --dim 4 --skiprerun --verbose --threads -1 > pca-diabetes-clustering.log 2>&1
python run_experiment.py --rp  --wine --dim 9 --skiprerun --verbose --threads -1 > rp-wine-clustering.log  2>&1
python run_experiment.py --rp  --diabetes --dim 10 --skiprerun --verbose --threads -1 > rp-diabetes-clustering.log  2>&1
python run_experiment.py --rf  --wine --dim 9 --skiprerun --verbose --threads -1 > rf-wine-clustering.log  2>&1
python run_experiment.py --rf  --diabetes --dim 14 --skiprerun --verbose --threads -1 > rf-diabetes-clustering.log  2>&1
