import numpy as np
from PIL import Image
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
from phasepack.phasecong import phasecong as pc

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def PSNR(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x**2 + y**2 + constant

    return numerator / denominator


def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx**2 + scharry**2)

def fsim(imageA, imageB):
    '''Adapted from
    https://github.com/up42/image-similarity-measures/blob/master/image_similarity_measures/quality_metrics.py'''
    T1 = 0.85
    T2 = 160.0
    alpha = (
        beta
    ) = 1

    pc1_dim = pc(imageA, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
    pc2_dim = pc(imageB, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
    pc1_2dim_sum = np.zeros((imageA.shape[0], imageA.shape[1]), dtype=np.float64)
    pc2_2dim_sum = np.zeros(
        (imageA.shape[0], imageB.shape[1]), dtype=np.float64
    )

    for orientation in range(6):
        pc1_2dim_sum += pc1_dim[4][orientation]
        pc2_2dim_sum += pc2_dim[4][orientation]

    gm1 = _gradient_magnitude(imageA, cv2.CV_16U)
    gm2 = _gradient_magnitude(imageB, cv2.CV_16U)

    # Calculate similarity measure for PC1 and PC2
    S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
    # Calculate similarity measure for GM1 and GM2
    S_g = _similarity_measure(gm1, gm2, T2)

    S_l = (S_pc ** alpha) * (S_g ** beta)

    numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
    return (numerator / denominator)
def compare_image(imageA, imageB, type):
    if type == 'ssim':
        s = ssim(imageA, imageB)
    elif type == 'mse':
        s = mse(imageA, imageB)
    elif type == 'psnr':
        s = PSNR(imageA, imageB)
    elif type == 'fsim':
        s = fsim(imageA, imageB)
    return s

def convert_img(img):
    im = Image.open(img)
    im = im.convert('L')
    im = np.array(im)
    im = im.astype(np.uint8)
    return im