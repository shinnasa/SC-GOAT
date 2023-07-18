#!/bin/sh
#SBATCH --partition gpu
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu 25G
#SBATCH --mail-user=s.nakamura.sakai@yale.edu
#SBATCH --mail-type=ALL

module load miniconda
conda activate sdv

python code/SupervisedSynthesizer.py adult CTGAN
python code/SupervisedSynthesizer.py adult TVAE
python code/SupervisedSynthesizer.py adult CopulaGAN

echo "Done"