import os

import imageio

from Perona_Malik_Base import *
import glob
from utils import convert_img
import Image

types_of_noise = ['gaussian', 'shot', 's&p']
img_dir = '/home/zoe/ECE6560_ImageDenoising/Images/'
params = {
    'gaussian': {'iters': 60, 'k': 1},
    'shot': {'iters': 60, 'k': 1},
    's&p': {'iters': 60, 'k': 1}
}

for noise in types_of_noise:
    print('----- Running experiment for ' + noise + ' Noise -----')
    # For each type of noise, run Perona-Malik
    img_dir_noise = img_dir + noise + '/'
    # list all images
    imgs = glob.glob(img_dir_noise + '*')
    # results directory
    results_dir = '/home/zoe/ECE6560_ImageDenoising/Results/'
    results_dir += noise + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # iterate through all noisy images
    for img in imgs:
        file = img.split(noise)[-1][1:]
        converted_img = convert_img(img=img)
        current_params = params[noise]
        denoised = diffusion(img, iters=current_params['iters'], k=current_params['k'])
        save_file = results_dir + file
        denoised = denoised.astype('uint8')
        denoised = Image.fromarray(denoised)
        imageio.imwrite(save_file, denoised)