#!/bin/sh
#SBATCH --partition gpu
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task 12
#SBATCH --gpus 1
#SBATCH --mem-per-cpu 3G
#SBATCH --mail-user=s.nakamura.sakai@yale.edu
#SBATCH --mail-type=ALL
# module load miniconda
# conda activate sdv
#encode balanced tuned
# Untuned
# Without Target Encoder
# Adult Data Set
# Data Num_iter Encode Balanced Tuned
python3.10 code/OptimizationApproach_XGBoost.py adult 100 False False False

# With Target Encoder
# Adult Data Set
python3.10 code/OptimizationApproach_XGBoost.py adult 100 True False False

# # Balanced Credit Card Data Set
python3.10 code/OptimizationApproach_XGBoost.py credit_card 100 False True False

# # Unbalanced Credit Card Data Set
python3.10 code/OptimizationApproach_XGBoost.py credit_card 100 False False False


# Tuned
# Without Target Encoder
# Adult Data Set
# python code/OptimizationApproach_XGBoost.py adult 20000 False False True

# # Unbalanced Credit Card Data Set (need files)
# python code/OptimizationApproach_XGBoost.py credit_card 100 False False True

# # Balanced Credit Card Data Set
# python code/OptimizationApproach_XGBoost.py credit_card 20000 False True True

# With Target Encoder 
# Adult Data Set (need files)
# python code/OptimizationApproach_XGBoost.py adult 20000 True False True