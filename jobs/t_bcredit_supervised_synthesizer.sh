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

python code/SupervisedSynthesizer.py balanced_credit_card CTGAN True False
python code/SupervisedSynthesizer.py balanced_credit_card TVAE True False
python code/SupervisedSynthesizer.py balanced_credit_card CopulaGAN True False

echo "Done"