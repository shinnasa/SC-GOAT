#echo "Script executed from: ${PWD}"

# Without Target Encoder
# Adult Data Set
# python3.10 ../code/BaselineModel.py adult 100 False False
# # Credit Card Data Set Unbalanced
# python3.10 ../code/BaselineModel.py credit_card 1000 False False
# # Credit Card Data Set Balanced
# python3.10 ../code/BaselineModel.py credit_card 100 False True

# With Target Encoder
# Adult Data Set
python3.10 ../code/BaselineModel.py adult 100 True False 
# Credit Card Data Set Unbalanced
python3.10 ../code/BaselineModel.py credit_card 100 True False
# Credit Card Data Set Balanced
python3.10 ../code/BaselineModel.py credit_card 100 True True