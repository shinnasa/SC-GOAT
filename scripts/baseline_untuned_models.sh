#encode balanced tuned
# Untuned
# Without Target Encoder
# Adult Data Set
python3.10 ../code/BaselineModel.py adult 100 False False False

# # Unbalanced Credit Card Data Set
python3.10 ../code/BaselineModel.py credit_card 100 False False False

# # Balanced Credit Card Data Set
# python3.10 ../code/BaselineModel.py credit_card 100 False True False

# With Target Encoder
# Adult Data Set
# python3.10 ../code/BaselineModel.py adult 100 True False False