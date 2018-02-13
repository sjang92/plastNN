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

The training data contains the following data types for 205 positive-label (apicoplast) proteins and 451 negative-label (non-apicoplast) proteins:
1. Protein sequence (positive.txt and negative.txt)
2. Position of the first nucleotide after the end of the signal peptide, predicted by signalP3.0 (1) (pos_tp.txt and neg_tp.txt)
3. Transcript levels corresponding to each protein at 8 time points, from Bartfai et al. (2) (pos_rna.txt and neg_rna.txt)

The unlabeled data contains similar files for 450 unlabeled proteins.
Both data can be found in the [data](https://github.com/sjang92/plastNN/tree/master/data) directory.

## Featurization

For each protein, plastNN constructs a feature vector of length 28. The first 20 elements represent fequencies of the 20 canonical amino acids in a 50-amino acid region immediately after the predicted signal peptide, and the next 8 elements are transcript levels at 8 time points. These vectors are used as input to the neural network.

## Model

PlastNN is a simple fully-connected neural network with 3 hidden layers, with each layer having 64, 64 and 16 output neurons respectively.

## Training and Evaluation

Neural networks were trained using the RMSProp optimization algorithm with a learning rate of 0.0001. 

## Results

The results are described in the following paper:

Insert Biorxiv link when available

##References
1. Nielsen, H., 2017. Predicting Secretory Proteins with SignalP. Protein Function Prediction: Methods and Protocols, pp.59-73.
2. Bártfai R, Hoeijmakers WAM, Salcedo-Amaya AM, Smits AH, Janssen-Megens E, Kaan A, et al. (2010) H2A.Z Demarcates Intergenic Regions of the Plasmodium falciparum Epigenome That Are Dynamically Marked by H3K9ac and H3K4me3. PLoS Pathog 6(12): e1001223. https://doi.org/10.1371/journal.ppat.1001223