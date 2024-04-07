import numpy as np
from PIL import Image
def convert_img(img):
    im = Image.open(img).convert('L')
    im = np.array(im)
    # normalize
    #im = (im - im.min()) / (im.max() - im.min())
    return im