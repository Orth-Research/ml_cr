from keras.models import Sequential
from keras import optimizers
from keras import models
from keras import layers
from keras.layers import BatchNormalization
from keras.layers import Dropout

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import metrics

def build_model(channels, outputs):
	"""
    Creates a convolutional neural network for a specific point group. 
	Args:
		channels: number of channels in the input data
		outputs: number of Stevens coefficients to predict

    Returns:
        Model for the CNN for the specific point group
    """
	model = Sequential()

	model.add(layers.Conv1D(128, 10, activation='relu', input_shape=(64, channels)))
	model.add(layers.Conv1D(256, 10, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling1D(2))

	model.add(layers.Conv1D(384, 10, activation='relu'))
	model.add(layers.Conv1D(256, 10, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.GlobalAveragePooling1D())

	model.add(layers.Dense(2048, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.4))

	model.add(layers.Dense(2048, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.4))

	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.4))

	model.add(layers.Dense(1024, activation='relu'))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.4))

	model.add(layers.Dense(outputs))
	
	model.compile(optimizer='adam', loss='mse', metrics=['mae'])
	return model
	
class new_callback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if (logs.get('val_mae')<0.35):
			self.model.stop_training = True
			
if __name__=='__main__':
	group = 'Oh'
	J = '4'
	
	dir = './TrainingData_{}_{}/'.format(group, J)
	data = dir + 'generated_data_{}.csv'
	targets = dir + 'generated_targets_{}.csv'
	
	x_train = np.array(pd.read_csv(data.format(100000), header=None))
	y_train = np.array(pd.read_csv(targets.format(100000), header=None))
	channels = len(x_train[0])//64

	x_test = np.array(pd.read_csv(data.format(30000), header=None))
	y_test = np.array(pd.read_csv(targets.format(30000), header=None))
	
	x_val = x_test[:15000]
	x_test = x_test[15000:]
	y_val = y_test[:15000]
	y_test = y_test[15000:]
	
	x_train = np.reshape(x_train, (-1, 64, channels))
	x_test = np.reshape(x_test, (-1, 64, channels))
	x_val = np.reshape(x_val, (-1, 64, channels))
	
	# center the image data for each channel (mean of zero)
	x_mean = np.mean(x_train, axis=(0,1), keepdims=True)
	x_train -= x_mean
	x_val -= x_mean
	x_test -= x_mean
	
	model = build_model(x_train.shape[2], len(y_train[0]))
	callbacks = new_callback()
	
	history = model.fit(x_train,
		y_train,
		epochs=45,
		batch_size=64,
		validation_data=(x_val, y_val),
		verbose=2,
		callbacks=[callbacks])
		
	y_real = np.array(pd.read_csv(targets.format(1000), header=None))
	x_real = np.array(pd.read_csv(data.format(1000), header=None))
	
	x_real = np.reshape(x_real, (-1, 64, channels))
	x_real -= x_mean
	y_pred = model.predict(x_real)
	
	with open(dir + 'results_{}_{}_1D.txt'.format(group, J), 'w') as text_file:
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
	
	model.save(dir + '{}_{}_1D_model.h5'.format(group, J))
		
	