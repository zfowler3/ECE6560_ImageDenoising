import glob
import numpy as np
import os
from PIL import Image
from utils import convert_img
import imageio
from scipy import ndimage


def gaussian_noise(input_img, mean, var):
    noisy_img = ndimage.gaussian_filter(input=input_img, sigma=var)
    #noisy_img = cv2.GaussianBlur(input_img, (8,8), 0)
    return noisy_img
def salt_and_pepper(input_img):
    noisy_img = np.copy(input_img)
    dims = int(input_img.shape[0] * input_img.shape[1])
    num_pixels_salt = np.random.randint(int(dims/100), int(dims/10))
    # salt
    for s in range(num_pixels_salt):
        row = np.random.randint(0, input_img.shape[0]-1)
        col = np.random.randint(0, input_img.shape[-1]-1)
        noisy_img[row, col] = 255
    num_pixels_pepper = np.random.randint(int(dims/100), int(dims/10))
    # pepper
    for p in range(num_pixels_pepper):
        row = np.random.randint(0, input_img.shape[0]-1)
        col = np.random.randint(0, input_img.shape[-1]-1)
        noisy_img[row, col] = 0
    return noisy_img

def shot_noise(input_img):
    noisy = np.random.poisson(input_img / 255.0 * input_img.max()) / input_img.max() * 255
    noisemap = np.ones((input_img.shape[0], input_img.shape[1])) * noisy.mean()
    noisy_img = noisy+np.random.poisson(noisemap)
    return noisy_img

img_folder = '/home/zoe/ECE6560_ImageDenoising/Images/'
all_imgs = glob.glob(img_folder + '*')

noises = ['gaussian', 's&p', 'shot']

for noise in noises:
    # apply noise to all imgs
    for img in all_imgs:
        if '.' in img:
            file_name = img
            updated_name = file_name.split('.')[0] + '_noise.' + file_name.split('.')[-1]
            updated_folder = updated_name.split('Images')[0] + 'Images/' + noise + '/'
            updated_name = updated_folder + updated_name.split('Images')[-1][1:]
            if not os.path.exists(updated_folder):
                os.makedirs(updated_folder)
            converted_img = convert_img(img)
            if noise == 'gaussian':
                noisy_img = gaussian_noise(input_img=converted_img, mean=10, var=3)
            elif noise == 's&p':
                noisy_img = salt_and_pepper(input_img=converted_img)
            else:
                noisy_img = shot_noise(input_img=converted_img)

            # Save image
            print(noisy_img.dtype)
            noisy_img = noisy_img.astype('uint8')
            noisy_img = Image.fromarray(noisy_img)
            imageio.imwrite(updated_name, noisy_img)
