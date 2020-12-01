import numpy as np
import pandas as pd
from numpy import savez_compressed
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
	parser = argparse.ArgumentParser()
    # Command line arguments
	parser.add_argument("input_dir", type=str, help="Input directory")
	parser.add_argument("output_dir", type=str, help="Ouptut directory")
	parser.add_argument("-s", "--save_mean", type=bool, default=True, help="True to save mean and std for x, y")
	
	args = parser.parse_args()
	INPUT_DIR = args.train_dir
	OUTPUT_DIR = args.output_dir
	SAVE = args.save_mean
	
	x = np.array(pd.read_csv(os.path.join(INPUT_DIR, "generated_data.csv"), header=None))
	y = np.array(pd.read_csv(os.path.join(INPUT_DIR, "generated_targets.csv"), header=None))
	channels = len(x[0])//64

	x = cwt(x, channels=channels)

	# center the image data for each channel (mean of zero)
	x_mean = np.mean(x, axis=(0,1,2), keepdims=True)
	
	# normalize each of the targets (mean of zero and std of one)
	y_mean = np.mean(y, axis=(0,), keepdims=True)
	y_std = np.std(y, axis=(0,), keepdims=True)

	if (SAVE):
		np.save(os.path.join(OUTPUT_DIR, "x_mean.npy"), x_mean)
		np.save(os.path.join(OUTPUT_DIR, "y_mean.npy"), y_mean)
		np.save(os.path.join(OUTPUT_DIR, "y_std.npy"), y_std)
	
	savez_compressed(os.path.join(OUTPUT_DIR, "generated_data_cwt.npz"), x)
	savez_compressed(os.path.join(OUTPUT_DIR, "generated_targets_cwt.npz"), y)