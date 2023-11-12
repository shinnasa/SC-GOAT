# SC-GOAT
This is the implementation of a **s**upervised and **g**eneartive **o**ptimization **a**ppraoch for **t**abular data called (SC-GOAT).

The framework integrates a supervised component tailored to the specific downstream task and employs a meta-learning approach to learn the optimal mixture distribution of existing synthetic distributions.

## One-time setup
Create python virtual environment scgoat_env:

```console
python -m venv scgoat_env
```

Activate python virtual environment scgoat_env: 

```console
source scgoat_env/bin/activate  
```

Install required python packages:

```console
python -m pip install -r requirements.txt
```

## Running
### Supervised Synthesizers

To test our method, first you need to generate synthetic data using the four generative methods, namely Gaussian Copula, CTGAN, Copula GAN, and TVAE, available from the Synthetic Data Vaul (SDV) python package, please use the script:

```console
SupervisedSynthesizer.py
```

To see the meaning of each argument:

```console
python code/SupervisedSynthesizer.py -h
```

Here is a simple example:

```console
python code/SupervisedSynthesizer.py -o data_test -i 300 -m TVAE -d adult -e False
```

### SC-GOAT

To generate the synthetic data using SC-GAOT, please use the script:

```console
Run_Experiments.py
```

To see the meaning of each argument:

```console
python code/Run_Experiments.py -h 
```

Here is a simple example:

```console
python code/Run_Experiments.py -o data_test -i 300 -d adult -e False -g 10000 -id 1
```

### Instructions for for reproducing an experiment

```console
python code/SupervisedSynthesizer.py -o data_test -i 300 -m CopulaGAN -d adult -e False
python code/SupervisedSynthesizer.py -o data_test -i 300 -m CTGAN -d adult -e False
python code/SupervisedSynthesizer.py -o data_test -i 300 -m GaussianCopula -d adult -e False
python code/SupervisedSynthesizer.py -o data_test -i 300 -m TVAE -d adult -e False
python code/Run_Experiments.py -o data_test -i 300 -d adult -e False -g 10000 -id 1
```
