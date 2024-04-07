import glob
import numpy as np
import os

def gaussian_noise(input_img, mean, var):
    gauss = np.random.normal(loc=mean, scale=var, size=(input_img.shape))

    return noisy_img

img_folder = '/home/zoe/ECE6560_ImageDenoising/Images/'
noise_folder = '/home/zoe/ECE6560_ImageDenoising/Images/Noise/'
# Create folder if not already done
if not os.path.exists(noise_folder):
    os.makedirs(noise_folder)

