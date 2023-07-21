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
# dataset method encode baseline
python code/SupervisedSynthesizer.py adult CTGAN True True
python code/SupervisedSynthesizer.py adult TVAE True True
python code/SupervisedSynthesizer.py adult CopulaGAN True True

python code/SupervisedSynthesizer.py adult CTGAN False True
python code/SupervisedSynthesizer.py adult TVAE False True
python code/SupervisedSynthesizer.py adult CopulaGAN False True

python code/SupervisedSynthesizer.py balanced_credit_card CTGAN False True
python code/SupervisedSynthesizer.py balanced_credit_card TVAE False True
python code/SupervisedSynthesizer.py balanced_credit_card CopulaGAN False True

python code/SupervisedSynthesizer.py unbalanced_credit_card CTGAN False True
python code/SupervisedSynthesizer.py unbalanced_credit_card TVAE False True
python code/SupervisedSynthesizer.py unbalanced_credit_card CopulaGAN False True

echo "Done"