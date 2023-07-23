#encode balanced tuned
# Untuned
# Without Target Encoder
# Adult Data Set
# Data Num_iter Encode Balanced Tuned
python3 code/OptimizationApproach_XGBoost.py adult 500 False False False

# # Unbalanced Credit Card Data Set
python3 code/OptimizationApproach_XGBoost.py credit_card 500 False False False

# # Balanced Credit Card Data Set
python3 code/OptimizationApproach_XGBoost.py credit_card 500 False True False

# With Target Encoder
# Adult Data Set
python3 code/OptimizationApproach_XGBoost.py adult 500 True False False

# Tuned
# Without Target Encoder
# Adult Data Set
python3 code/OptimizationApproach_XGBoost.py adult 500 False False True

# # Unbalanced Credit Card Data Set (need files)
python3 code/OptimizationApproach_XGBoost.py credit_card 500 False False True

# # Balanced Credit Card Data Set
python3 code/OptimizationApproach_XGBoost.py credit_card 500 False True True

# With Target Encoder 
# Adult Data Set (need files)
python3 code/OptimizationApproach_XGBoost.py adult 500 True False True