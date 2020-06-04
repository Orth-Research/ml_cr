import numpy as np
import pandas as pd
from numpy import savez_compressed
import pywt

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
    scales = np.arange(1, shape+1)

    data_cwt = np.ndarray(shape=(size, shape, shape, channels), dtype=np.float16)
    for i in range(size):
        if (i % 1000 == 0):
            print('.', end='')

        # generating each of the channels
        for j in range(channels):
            signal = data[i][j*shape: shape+(j*shape)]
            coeff, freq = pywt.cwt(signal, scales, wavelet, 1)
            data_cwt[i, :, :, j] = coeff

    return data_cwt
	
	
if __name__=='__main__':
	group = 'Oh'
	J = '4'
	
	dir = '/TrainingData_{}_{}/'.format(group, J)
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

	x_val = cwt(x_val, channels=channels)
	x_train = cwt(x_train, channels=channels)
	x_test = cwt(x_test, channels=channels)

	# center the image data for each channel (mean of zero)
	x_mean = np.mean(x_train, axis=(0,1,2), keepdims=True)
	#x_train -= x_mean
	#x_val -= x_mean
	#x_test -= x_mean
	
	# normalize each of the targets (mean of zero and std of one)
	y_mean = np.mean(y_train, axis=(0,), keepdims=True)
	#y_train -= y_mean
	#y_val -= y_mean
	#y_test -= y_mean
	y_std = np.std(y_train, axis=(0,), keepdims=True)
	#y_train /= y_std
	#y_val /= y_std
	#y_test /= y_std
	
	np.save(dir + 'x_mean_{}_{}.npy'.format(group, J), x_mean)
	np.save(dir + 'y_mean_{}_{}.npy'.format(group, J), y_mean)
	np.save(dir + 'y_std_{}_{}.npy'.format(group, J), y_std)
	
	savez_compressed(dir + 'x_train_{}_{}.npz'.format(group, J), x_train)
	savez_compressed(dir + 'x_test_{}_{}.npz'.format(group, J), x_test)
	savez_compressed(dir + 'x_val_{}_{}.npz'.format(group, J), x_val)
	savez_compressed(dir + 'y_train_{}_{}.npz'.format(group, J), y_train)
	savez_compressed(dir + 'y_test_{}_{}.npz'.format(group, J), y_test)
	savez_compressed(dir + 'y_val_{}_{}.npz'.format(group, J), y_val)