#echo "Script executed from: ${PWD}"
# Adult Data Set
python3.10 ../code/BaselineModel.py adult 100 False
# Credit Card Data Set Unbalanced
python3.10 ../code/BaselineModel.py credit_card 100 False
# Credit Card Data Set Balanced
python3.10 ../code/BaselineModel.py credit_card 100 True