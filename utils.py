import numpy as np
import PIL
def convert_img(img):
    im = PIL.Image.open(img).convert('L')
    im = np.array(im)
    return im