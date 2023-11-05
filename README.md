# SC-GOAT
This is the implementation of a **s**upervised and **g**eneartive **o**ptimization **a**ppraoch for **t**abular data called (SC-GOAT).

The framework integrates a supervised component tailored to the specific downstream task and employs a meta-learning approach to learn the optimal mixture distribution of existing synthetic distributions.

## One-time setup
TODO Include a script to install all the required packages automatically.

## Running
### Supervised Synthesizers

To test our method, first you need to generate synthetic data using the four generative methods, namely Gaussian Copula, CTGAN, Copula GAN, and TVAE, available from the Synthetic Data Vaul (SDV) python package, please use the script:
```console
python code/SupervisedSynthesizer.py DATA_SET_NAME METHOD_NAME ENCODE
```
DATA_SET_NAME can be ['adult', 'unbalanced_credit_card', 'balanced_credit_card']

METHOD_NAME can be ['CopulaGAN', 'CTGAN', 'GaussianCopula', 'TVAE']

ENCODE can be [True, False]

To generate the synthetic data using SC-GAOT, then please use the script:

```console
python code/Run_Experiments.py DATA_SET_NAME ITR ENCODE BALANCED EXPERIMENT_ID 
```
DATA_SET_NAME can be ['adult', 'credit_card']

ITR can be any positive integer

ENCODE can be [True, False]

BALANCED can be [True, False]

EXPERIMENT_ID can be any positive integer


Here is a simple example:

```console
python code/SupervisedSynthesizer.py balanced_credit_card CopulaGAN False
python code/SupervisedSynthesizer.py balanced_credit_card CTGAN False
python code/SupervisedSynthesizer.py balanced_credit_card GaussianCopula False
python code/SupervisedSynthesizer.py balanced_credit_card TVAE False
python code/Run_Experiments.py credit_card 350 False True 1 
```

TODO Change the scripts to see the meaning of each argument.

TODO Add argument for the output directory path.
