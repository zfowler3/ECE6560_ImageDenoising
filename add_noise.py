import glob
from random import random

import numpy as np
import os

def gaussian_noise(input_img, mean, var):
    gauss = np.random.normal(loc=mean, scale=var, size=input_img.shape)
    noisy_img = gauss + input_img
    return noisy_img
def salt_and_pepper(input_img):
    num_pixels = random.randint(int(input_img.shape[0]/4), int(input_img.shape[0]/2))

    return noisy_img

img_folder = '/home/zoe/ECE6560_ImageDenoising/Images/'
noise_folder = '/home/zoe/ECE6560_ImageDenoising/Images/Noise/'
# Create folder if not already done
if not os.path.exists(noise_folder):
    os.makedirs(noise_folder)

