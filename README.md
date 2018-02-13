# PlastNN

This is the public repository for PlastNN, a neural-network based predictor for Apicoplast proteins.

## How to Run
PlastNN was developed and tested on Ubuntu 16.04, Bash on Ubuntu on Windows, and MacOSX using Python3.
We've also checked that it runs fine on windows with Visual Studio 2017, but we are not providing
tools to run the script on this specific environment.

To setup the python environment and install dependencies, clone the repo and run:
'''
> source setup.sh
'''

Once this is done, you should find yourself in the virtual environment named 'venv'.

To train the model with default parameters (recommended) and produce output, run:
'''
> ./train.sh
'''

By default, this script uses 6-fold cross-validation to train 6 fully-connected models.
Once training is over, these 6 models then vote on the unlabeled protein datapoints to assign labels.

You should see results outputted in the newly created 'results' directory. 'perf.csv' contains the evaluation results
obtained after each epoch, and 'vote.csv' contains the final voting results made by 6 different models
trained using 6-fold cross validation.

If you want to run the script with different hyperparameters (learning rate, number of layers and neurons, etc),
check the tensorflow app flags defined in 'src/trainer.py' and re-run accordingly.

## Background

## Data
Explain data here (rna_sequence, tp start indices, and transcriptome data)
The training data contains 205 positive-label (apicoplast) proteins and 451 negative-label (non-apicoplast) proteins.

The unlabeled data contains
Both data can be found in the [data](https://github.com/sjang92/plastNN/tree/master/data) directory.

## Featurization

Explain how the raw data is featurized into vectors of length 28.

## Model

PlastNN is a simple 3 layer fully-connected neural network, with each layer having 64, 64 and 16 output neurons respectively.

## Training and Evaluation

The model was trained using

## Results
