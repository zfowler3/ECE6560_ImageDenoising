import glob
from random import random

import numpy as np
import os

def gaussian_noise(input_img, mean, var):
    gauss = np.random.normal(loc=mean, scale=var, size=input_img.shape)
    noisy_img = gauss + input_img
    return noisy_img
def salt_and_pepper(input_img):
    noisy_img = np.copy(input_img)
    num_pixels_salt = random.randint(int(input_img.shape[0]/10), int(input_img.shape[0]/4))
    # salt
    for s in range(num_pixels_salt):
        row = random.randint(0, input_img.shape[0]-1)
        col = random.randint(0, input_img.shape[-1]-1)
        noisy_img[row, col] = 255
    num_pixels_pepper = random.randint(int(input_img.shape[0]/10), int(input_img.shape[0]/4))
    # pepper
    for p in range(num_pixels_pepper):
        row = random.randint(0, input_img.shape[0]-1)
        col = random.randint(0, input_img.shape[-1]-1)
        noisy_img[row, col] = 0

    return noisy_img

img_folder = '/home/zoe/ECE6560_ImageDenoising/Images/'
noise_folder = '/home/zoe/ECE6560_ImageDenoising/Images/Noise/'
# Create folder if not already done
if not os.path.exists(noise_folder):
    os.makedirs(noise_folder)

