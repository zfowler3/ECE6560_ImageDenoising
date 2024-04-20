import glob

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

from utils import *


def custom_coeff(I, k=0.1):
    inside = ((I*np.sqrt(3))/(k*np.sqrt(2)))**2
    return (0.04+0.2*np.exp(-inside))

def diffusion_coeff1(I, k=0.1):
    # exponential
    inside = (I**2)/(k**2)
    return np.exp(-inside)

def diffusion_coeff2(I, k=0.1):
    denom = 1 + (np.power(I, 2)/np.power(k, 2))
    return 1/denom
def coeff_vs_grad():
    results_list = []
    results_list_1 = []
    I = np.arange(0, 3, step=0.005)
    for val in I:
        r = custom_coeff(val)
        rr = diffusion_coeff1(val)
        results_list.append(r)
        results_list_1.append(rr)

    results_list = np.array(results_list)
    plt.plot(I, results_list, label='Coeff3')
    plt.plot(I, np.array(results_list_1), label='Coeff2')
    plt.xlabel('Gradient of I')
    plt.ylabel('Output of Diffusion Coefficient')
    plt.title('Comparison of Diffusion Coefficients')
    plt.legend()
    plt.grid()
    plt.show()
    return

def return_metrics():
    coeffs = ['coeff1', 'coeff3']
    noises = ['gaussian', 'shot', 's&p']
    metrics = ['ssim', 'psnr', 'fsim', 'mse']
    img_dir = '/home/zoe/ECE6560_ImageDenoising/Images/Clean/'
    results_ = '/home/zoe/ECE6560_ImageDenoising/Results/'
    imgs = glob.glob(img_dir + '*')
    for c in coeffs:
        results_dir = results_ + c + '/'
        print('-------------------------')
        print('COEFFICIENT: ', c)
        for i in imgs:
            img = convert_img(i)
            img_name = i.split('/')[-1].split('.')[0]
            ext = i.split('/')[-1].split('.')[-1]
            print('Image: ', img_name)
            for noise in noises:
                print('*****')
                print('Noise: ', noise)
                result_img_name = results_dir + noise + '/' + img_name + '_noise.' + ext
                result_img = convert_img(result_img_name)
                for m in metrics:
                    metric_ = compare_image(img, result_img, type=m)
                    print('Metric ' + m + ' ... ' + str(metric_))

    return

#coeff_vs_grad()
return_metrics()