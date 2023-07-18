#!/bin/sh
#SBATCH --partition week
#SBATCH --time 48:00:00
#SBATCH --cpus-per-task 24
#SBATCH --mem-per-cpu 5G
#SBATCH --mail-user=s.nakamura.sakai@yale.edu
#SBATCH --mail-type=ALL

module load miniconda
conda activate sdv

python code/SupervisedSynthesizer.py unbalanced_credit_card CTGAN
python code/SupervisedSynthesizer.py unbalanced_credit_card TVAE
python code/SupervisedSynthesizer.py unbalanced_credit_card CopulaGAN

echo "Done"