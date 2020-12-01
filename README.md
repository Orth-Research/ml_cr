# Crystal-Field-Machine-Learning

This repository holds the code for the paper ["Learning crystal field parameters using convolutional neural networks"](https://arxiv.org/pdf/2011.12911.pdf) authored by Noah F. Berthusen, Yuriy Sizyuk, Mathias S. Scheurer, and Peter P. Orth. This work presents a deep machine learning algorithm that uses a two-dimensional convolutional neural network (CNN) to determine crystal field Stevens parameters from thermodynamic observables, given a total angular momentum J of the ground state multiplet and crystal field site symmetry group G.

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
    * Specify all of the required data in ```DataGeneration.py```. This includes temperature and magnetic field ranges for specific heat, susceptibility, and magnetization, magnetic field lattice directions, and of course the point group symmetry G and the angular momentum quantum numbers J (total angular momentum), L (total orbital angular momentum), and S (total spin) of the ground state multiplet. The program can be run entirely with command line arguments. View the command line arguments with the following command 
      ```
      python DataGeneration.py -h
      ```
      The required variables could also be hard-coded from within the program if working with the command line arguments is cumbersome. The output directory can be specified, and the data will be saved as ```OUTPUT_DIR/generated_data.csv```, and the target Stevens parameters will be saved as ```OUTPUT_DIR/generated_targets.csv```. 
    * Generate training data in amounts suitable to be trained on.
    * Using command line arguments, specify the input and output directories for ```WaveletTransform.py``` to transform the 1D thermodynamic observable data into the 2D scaleogram format that is needed for the CNN architecture. The program expects the input data to have the filename ```INPUT_DIR/generated_data.csv```. In this file, you can choose to save the mean and standard deviations for the generated_data/targets. This is needed while training and when evaluating any predictions after training.

3. Train model
   * The following is best done on a GPU device. After specifying the location of CWT data generated in step 2 through command line arguments, run ```StevensTraining.py``` to train the CNN. Specify training, validation, and out directories to load the training/validation data, and to save the trained model. The number of epochs, batch size, and early stopping threshold can also be specified.
