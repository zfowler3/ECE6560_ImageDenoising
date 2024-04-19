import os

import imageio
import matplotlib.pyplot as plt

from Perona_Malik_Base import *
import glob
from utils import convert_img
from PIL import Image

types_of_noise = ['gaussian', 'shot', 's&p']
img_dir = '/home/zoe/ECE6560_ImageDenoising/Images/'
params = {
    'gaussian': {'iters': 30, 'k': .1},
    'shot': {'iters': 30, 'k': .1},
    's&p': {'iters': 30, 'k': .1}
}
coeff = 1
for noise in types_of_noise:
    print('----- Running experiment for ' + noise + ' Noise -----')
    # For each type of noise, run Perona-Malik
    img_dir_noise = img_dir + noise + '/'
    # list all images
    imgs = glob.glob(img_dir_noise + '*')
    # results directory
    results_dir = '/home/zoe/ECE6560_ImageDenoising/Results/X_' + str(coeff) + '/'
    results_dir += noise + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # iterate through all noisy images
    for img in imgs:
        file = img.split(noise)[-1][1:]
        converted_img = convert_img(img=img)
        current_params = params[noise]
        # first
        #denoised = diffusion_equation(converted_img, iters=current_params['iters'], k=current_params['k'])
        #second
        denoised = diffusion_four_directions(converted_img, iters=current_params['iters'], k=current_params['k'], coeff=coeff)
        save_file = results_dir + file
        plt.imshow(denoised, cmap='gray')
        #plt.show()
        plt.imsave(save_file, denoised, cmap='gray')