from keras.models import Sequential
from keras import optimizers
from keras import models
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Dropout

import tensorflow as tf
import numpy as np
import pandas as pd
import pywt
from sklearn import metrics
from sklearn.metrics import classification_report


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

class new_callback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if (logs.get('val_mae')<0.26):
			self.model.stop_training = True
	
if __name__=='__main__':
	group = 'D3h'
	J = '4'
	W_sign = -1
	
	dir = './TrainingData_{}_{}/TrainingData_{}_{}_{}/'.format(group, J, group, J, W_sign)
	data = dir + 'generated_data_{}.csv'
	targets = dir + 'generated_targets_{}.csv'
	
	x_train = np.load(dir + 'x_train_{}_{}.npz'.format(group, J))['arr_0']
	x_test = np.load(dir + 'x_test_{}_{}.npz'.format(group, J))['arr_0']
	x_val = np.load(dir + 'x_val_{}_{}.npz'.format(group, J))['arr_0']
	
	y_train = np.load(dir + 'y_train_{}_{}.npz'.format(group, J))['arr_0']
	y_test = np.load(dir + 'y_test_{}_{}.npz'.format(group, J))['arr_0']
	y_val = np.load(dir + 'y_val_{}_{}.npz'.format(group, J))['arr_0']
	
	x_mean = np.load(dir + 'x_mean_{}_{}.npy'.format(group, J))
	y_mean = np.load(dir + 'y_mean_{}_{}.npy'.format(group, J))
	y_std = np.load(dir + 'y_std_{}_{}.npy'.format(group, J))
	
	# center the image data for each channel (mean of zero)
	x_train -= x_mean
	x_val -= x_mean
	x_test -= x_mean
	
	# normalize each of the targets (mean of zero and std of one)
	y_train -= y_mean
	y_val -= y_mean
	y_test -= y_mean
	y_train /= y_std
	y_val /= y_std
	y_test /= y_std
	
	model = build_model(x_train.shape[3], len(y_train[0]))
	callbacks = new_callback()

	history = model.fit(x_train,
		y_train,
		epochs=100,
		batch_size=64,
		validation_data=(x_val, y_val),
		verbose=2,
                callbacks=[callbacks])
		
		
	y_real = np.array(pd.read_csv(targets.format(1000), header=None))
	x_real = np.array(pd.read_csv(data.format(1000), header=None))
	
	x_real = cwt(x_real, channels=x_train.shape[3])
	x_real -= x_mean
	y_pred = model.predict(x_real)
	y_pred = (y_pred * y_std) + y_mean
	
	with open(dir + 'results_{}_{}.txt'.format(group, J), 'w') as text_file:
		print(model.summary(), file=text_file)
		
		print(model.evaluate(x_test, y_test, verbose=0), file=text_file)
		
		if (group == 'Oh'):
			num_coeff = len(y_pred[0])
		else:
			num_coeff = len(y_pred[0])-1		

		for i in range(num_coeff):
			print('Mean absolute error: {}'.format(metrics.mean_absolute_error(y_real[:,i], y_pred[:,i])), file=text_file)
			print('Mean squared error: {}'.format(metrics.mean_squared_error(y_real[:,i], y_pred[:,i])), file=text_file)
			print('Explained varience score: {}'.format(metrics.explained_variance_score(y_real[:,i], y_pred[:,i])), file=text_file)
			print('r^2 score: {}'.format(metrics.r2_score(y_real[:,i], y_pred[:,i])), file=text_file)
			print('', file=text_file)
		
		if (group != 'Oh'):
			last_real = [0 if i < 0 else 1 for i in y_real[:,len(y_pred[0])-1]]
			last_pred = [0 if i < 0 else 1 for i in y_pred[:,len(y_pred[0])-1]]
			report = classification_report(last_real, last_pred)
			print(report, file=text_file)	
	
	model.save(dir + '{}_{}_{}_model.h5'.format(group, J, W_sign))
