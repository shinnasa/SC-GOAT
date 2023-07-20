#encode balanced tuned
# Untuned
# Without Target Encoder
# Adult Data Set
python3.10 ../code/OptimizationApproach_XGBoost.py adult 100 False False False

# # Unbalanced Credit Card Data Set
python3.10 ../code/OptimizationApproach_XGBoost.py credit_card 100 False False False

# # Balanced Credit Card Data Set
python3.10 ../code/OptimizationApproach_XGBoost.py credit_card 100 False True False

# With Target Encoder
# Adult Data Set
python3.10 ../code/OptimizationApproach_XGBoost.py adult 100 True False False

# Tuned
# Without Target Encoder
# Adult Data Set
python3.10 ../code/OptimizationApproach_XGBoost.py adult 100 False False True

# # Unbalanced Credit Card Data Set (need files)
python3.10 ../code/OptimizationApproach_XGBoost.py credit_card 100 False False True

# # Balanced Credit Card Data Set
python3.10 ../code/OptimizationApproach_XGBoost.py credit_card 100 False True True

# With Target Encoder 
# Adult Data Set (need files)
python3.10 ../code/OptimizationApproach_XGBoost.py adult 100 True False True