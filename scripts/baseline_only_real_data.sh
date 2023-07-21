#echo "Script executed from: ${PWD}"
# Adult Data Set
python3 ../code/BaselineOnlyRealData.py adult 100 False False
# Credit Card Data Set Unbalanced
python3 ../code/BaselineOnlyRealData.py credit_card 100 False False
# Credit Card Data Set Balanced
python3 ../code/BaselineOnlyRealData.py credit_card 100 False True

# With Target Encoder
# Adult Data Set
python3 ../code/BaselineOnlyRealData.py adult 100 True False
