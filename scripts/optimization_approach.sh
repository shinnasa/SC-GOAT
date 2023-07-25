#encode balanced tuned
# Untuned
# Without Target Encoder
# Adult Data Set
# Data Num_iter Encode Balanced Tuned
python3 code/OptimizationApproach_XGBoost.py adult 100 False False False

# With Target Encoder
# Adult Data Set
python3 code/OptimizationApproach_XGBoost.py adult 100 True False False

# # Balanced Credit Card Data Set
python3 code/OptimizationApproach_XGBoost.py credit_card 100 False True False

# Unbalanced Credit Card Data Set
python3 code/OptimizationApproach_XGBoost.py credit_card 100 False False False

# Tuned
# Without Target Encoder
# Adult Data Set
# python3 code/OptimizationApproach_XGBoost.py adult 2000 False False True

# # Unbalanced Credit Card Data Set (need files)
# python3 code/OptimizationApproach_XGBoost.py credit_card 2000 False False True

# # Balanced Credit Card Data Set
# python3 code/OptimizationApproach_XGBoost.py credit_card 2000 False True True

# With Target Encoder 
# Adult Data Set (need files)
# python3 code/OptimizationApproach_XGBoost.py adult 2000 True False True