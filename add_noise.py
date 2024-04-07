import glob
from random import random
from skimage.util import random_noise
import numpy as np
import os
import PIL
from utils import convert_img


def gaussian_noise(input_img, mean, var):
    gauss = np.random.normal(loc=mean, scale=var, size=input_img.shape)
    noisy_img = gauss + input_img
    return noisy_img
def salt_and_pepper(input_img):
    noisy_img = np.copy(input_img)
    dims = int(input_img.shape[0] * input_img.shape[1])
    num_pixels_salt = random.randint(int(dims/100), int(dims/10))
    # salt
    for s in range(num_pixels_salt):
        row = random.randint(0, input_img.shape[0]-1)
        col = random.randint(0, input_img.shape[-1]-1)
        noisy_img[row, col] = 255
    num_pixels_pepper = random.randint(int(dims/100), int(dims/10))
    # pepper
    for p in range(num_pixels_pepper):
        row = random.randint(0, input_img.shape[0]-1)
        col = random.randint(0, input_img.shape[-1]-1)
        noisy_img[row, col] = 0

    return noisy_img
def shot_noise(input_img):
    noisy_img = random_noise(input_img, mode='poisson')
    return noisy_img

img_folder = '/home/zoe/ECE6560_ImageDenoising/Images/'
all_imgs = glob.glob(img_folder + '*')

noise_folder = '/home/zoe/ECE6560_ImageDenoising/Images/Noise/'
# Create folder if not already done
if not os.path.exists(noise_folder):
    os.makedirs(noise_folder)

noises = ['gaussian', 's&p', 'shot']

for noise in noises:
    # apply noise to all imgs
    for img in all_imgs:
        file_name = img
        updated_name = file_name.split('.')[0] + '_noise.' + file_name.split('.')[-1]
        updated_folder = updated_name.split('Images')[0] + 'Images/' + noise + '/'
        updated_name = updated_folder + updated_name.split('Images')[-1][1:]
        if not os.path.exists(updated_folder):
            os.makedirs(updated_folder)
        converted_img = convert_img(img)
        if noise == 'gaussian':
            noisy_img = gaussian_noise(input_img=converted_img, mean=0, var=0.01)
        elif noise == 's&p':
            noisy_img = salt_and_pepper(input_img=converted_img)
        else:
            noisy_img = shot_noise(input_img=converted_img)

        # Save image
        act_img = PIL.Image.fromarray(noisy_img)
        act_img.save(updated_name)
