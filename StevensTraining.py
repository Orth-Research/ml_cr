from keras import models
from keras import layers
from keras.layers import Dropout

import tensorflow as tf
import numpy as np
import pywt
import os
import argparse


def cwt(data, channels=6, wavelet='morl'):
    """
    Applies the continuous wavelet transformation on each dataset (susc, spec heat, magnetization at 4 temps) individually
    and combines each convolution into a multi-channel image to be fed to the CNN
    Args:
        data: 1D array containing each dataset
        channels: number of channels in the output image (susc, spec heat, magnetization at 4 temps)
        waveletname: mother wavelet function to be used Can be any from 
            https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html, specifically pywt.wavelist()

    Returns:
        Input data reshaped into a (shape x shape) image with 'channels' channels
    """
    shape = data.shape[1]//channels
    size = data.shape[0]
    scales = [1.09051, 1.18921, 1.29684, 1.41421, 1.54221, 1.68179, 1.83401, 2., \
		2.18102, 2.37841, 2.59368, 2.82843, 3.08442, 3.36359, 3.66802, 4., \
		4.36203, 4.75683, 5.18736, 5.65685, 6.16884, 6.72717, 7.33603, 8., \
		8.72406, 9.51366, 10.3747, 11.3137, 12.3377, 13.4543, 14.6721, 16., \
		17.4481, 19.0273, 20.7494, 22.6274, 24.6754, 26.9087, 29.3441, 32., \
		34.8962, 38.0546, 41.4989, 45.2548, 49.3507, 53.8174, 58.6883, 64.]

    data_cwt = np.ndarray(shape=(size, len(scales), shape, channels), dtype=np.float16)
    for i in range(size):
        if (i % 1000 == 0):
            print('.', end='')

        # generating each of the channels
        for j in range(channels):
            signal = data[i][j*shape: shape+(j*shape)]
            coeff, freq = pywt.cwt(signal, scales, wavelet, 1)
            data_cwt[i, :, :, j] = coeff

    return data_cwt
	
def build_model(channels, outputs):
	"""
    Creates a convolutional neural network for a specific point group. 
	Args:
		channels: number of channels in the input data
		outputs: number of Stevens coefficients to predict

    Returns:
        Model for the CNN for the specific point group
    """
	model = models.Sequential()

	model.add(layers.Conv2D(96, (3, 3), activation='relu', input_shape=(48, 64, channels)))
	model.add(layers.Conv2D(96, (3, 3), activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(layers.Conv2D(256, (3, 3), activation='relu'))
	model.add(layers.Conv2D(256, (3, 3), activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(layers.Flatten())

	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(Dropout(0.3))

	model.add(layers.Dense(2048, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(Dropout(0.3))

	model.add(layers.Dense(outputs))
	model.summary()
	print(model.summary())	
	model.compile(optimizer='adam', loss='mse', metrics=['mae'])
	return model
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
    # Command line arguments
	parser.add_argument("train_dir", type=str, help="Train directory")
	parser.add_argument("val_dir", type=str, help="Validation directory")
	parser.add_argument("output_dir", type=str, help="Ouptut directory")
	parser.add_argument("-e", "--epochs", type=int, default=100, help="Epochs")
	parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
	parser.add_argument("-s", "--early_stop", type=float, default=0.0, help="Stop training once val MAE falls below")
	
	args = parser.parse_args()
	TRAIN_DIR = args.train_dir
	VAL_DIR = args.val_dir
	OUTPUT_DIR = args.output_dir
	NUM_EPOCHS = args.epochs
	BATCH_SIZE = args.batch_size
	EARLY_STOP = args.early_stop
	
	x_train = np.load(os.path.join(TRAIN_DIR, "generated_data_cwt.npz"))['arr_0']
	x_val = np.load(os.path.join(VAL_DIR, "generated_data_cwt.npz"))['arr_0']
	
	y_train = np.load(os.path.join(TRAIN_DIR, "generated_targets_cwt.npz"))['arr_0']
	y_val = np.load(os.path.join(VAL_DIR, "generated_targets_cwt.npz"))['arr_0']
	
	x_mean = np.load(os.path.join(TRAIN_DIR, "x_mean.npy"))
	y_mean = np.load(os.path.join(TRAIN_DIR, "y_mean.npy"))
	y_std = np.load(os.path.join(TRAIN_DIR, "y_std.npy"))
	
	# center the image data for each channel (mean of zero)
	x_train -= x_mean
	x_val -= x_mean
	
	# normalize each of the targets (mean of zero and std of one)
	y_train -= y_mean
	y_val -= y_mean
	y_train /= y_std
	y_val /= y_std
	
	model = build_model(x_train.shape[3], len(y_train[0]))

	class new_callback(tf.keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs={}):
			if (logs.get('val_mae') < EARLY_STOP):
				self.model.stop_training = True
	callbacks = new_callback()

	history = model.fit(x_train,
		y_train,
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		validation_data=(x_val, y_val),
		verbose=2,
		callbacks=[callbacks]
	)
	
	model.save(os.path.join(OUTPUT_DIR, "model.h5"))
