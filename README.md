# Crystal-Field-Machine-Learning

This repository holds the code for the paper Learning crystal field parameters using convolutional neural networks authored by Noah F. Berthusen, Yuriy Sizyuk, Mathias S. Scheurer, and Peter P. Orth. This work aimed to use a two-dimensional convolutional neural network (CNN) to determine the Stevens parameters for a given angular momentum J and crystal field symmetry group G from thermodynamic observables.

This document will guide you through installing and running the code. 

## Training Pipeline
The network can be trained to predict the Stevens parameters for a given point group symmetry. Training the network to predict the coefficients for experimental examples can also be one. For both, a sizeable amount of training data needs to be generated.

1. Download repository and update your python environment.
   * If you are familiar with git, clone the repository with the following command:
     ```
     git clone git@github.com:Orth-Research/ml_cr.git
     ```
     Alternatively, on the top right page of this page, download the zipped version of the repository and unpack at your desired place.

2. Prepare your data
    * Specify all of the required data in ```DataGeneration.py```. This includes ranges for specific heat, susceptibility, and magnetization, high-symmetry direction, and of course point group symmetry and J, L, and S.
    * Generate training data in amounts suitable to be trained on.
    * Specify the input and output directories for ```WaveletTransform.py``` to transform the 1D thermodynamic observable data into the 2D scaleogram format that is needed for the CNN architecture. 

3. Train model
   * The following is best done on a GPU device. After specifying the location of CWT data generated in step 2, run ```StevensTraining.py``` to train the CNN.
