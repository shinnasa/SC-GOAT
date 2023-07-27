#!/bin/sh
#SBATCH --partition gpu
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 12
#SBATCH --gpus 1
#SBATCH --mem-per-cpu 3G
#SBATCH --mail-user=s.nakamura.sakai@yale.edu
#SBATCH --mail-type=ALL

module load miniconda
conda activate sdv
# data, opt_itr, encode, balanced, experiment
python code/Run_Experiments.py credit_card 150 False True $1


echo "Done"