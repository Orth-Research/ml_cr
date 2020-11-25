# Crystal-Field-Machine-Learning

This repository holds the code for the paper "Learning crystal field parameters using convolutional neural networks" authored by Noah F. Berthusen, Yuriy Sizyuk, Mathias S. Scheurer, and Peter P. Orth. This work presents a deep machine learning algorithm that uses a two-dimensional convolutional neural network (CNN) to determine crystal field Stevens parameters from thermodynamic observables, given a total angular momentum J of the ground state multiplet and crystal field site symmetry group G.

This document will guide you through installing and running the code. 

## Training Pipeline
The network can be trained to predict the Stevens parameters for a given point group symmetry. Training the network to predict the coefficients for experimental examples can also be done. For both, a sizeable amount of training data needs to be generated.

1. Download repository and update your python environment.
   * If you are familiar with git, clone the repository with the following command:
     ```
     git clone git@github.com:Orth-Research/ml_cr.git
     ```
     Alternatively, on the top right page of this page, download the zipped version of the repository and unpack at your desired place.

2. Prepare your data
    * Specify all of the required data in ```DataGeneration.py```. This includes temperature and magnetic field ranges for specific heat, susceptibility, and magnetization, magnetic field lattice directions, and of course the point group symmetry G and the angular momentum quantum numbers J (total angular momentum), L (total orbital angular momentum), and S (total spin) of the ground state multiplet.
    * Generate training data in amounts suitable to be trained on.
    * Specify the input and output directories for ```WaveletTransform.py``` to transform the 1D thermodynamic observable data into the 2D scaleogram format that is needed for the CNN architecture. 

3. Train model
   * The following is best done on a GPU device. After specifying the location of CWT data generated in step 2, run ```StevensTraining.py``` to train the CNN.
