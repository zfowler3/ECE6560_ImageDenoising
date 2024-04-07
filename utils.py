import numpy as np
from PIL import Image
def convert_img(img):
    im = Image.open(img)
    im = im.convert('L')
    im = np.array(im)
    im = im.astype(np.uint8)
    # normalize
    #im = (im - im.min()) / (im.max() - im.min())
    return im