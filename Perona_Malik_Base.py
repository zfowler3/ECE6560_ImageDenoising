import numpy as np

def diffusion_coeff1(I, k):
    # exponential
    inside = (I**2)/(k**2)
    return np.exp(-inside)

def diffusion_coeff2(I, k):
    denom = 1 + (np.power(I, 2)/np.power(k, 2))
    return 1/denom

def custom_coeff1(I, k):
    inside = ((I*np.sqrt(3))/(k*np.sqrt(2)))**2
    return 0.6*np.exp(-inside)

def custom_coeff(I, k):
    inside = ((I*np.sqrt(3))/(k*np.sqrt(2)))**2
    return (0.04+0.2*np.exp(-inside))


def diffusion_(img, iters, k, lam=0.12, coeff=1):
    #http://image.diku.dk/imagecanon/material/PeronaMalik1990.pdf
    img = img / 255
    img_new = np.zeros(img.shape, dtype=img.dtype)

    for step in range(iters):

        NORTH = img[:-2, 1:-1] - img[1:-1, 1:-1]

        SOUTH = img[2:, 1:-1] - img[1:-1, 1:-1]

        EAST = img[1:-1, 2:] - img[1:-1, 1:-1]

        WEST = img[1:-1, :-2] - img[1:-1, 1:-1]

        if coeff == 1:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 (diffusion_coeff1(NORTH, k) * NORTH +
                                        diffusion_coeff1(SOUTH, k) * SOUTH +
                                        diffusion_coeff1(EAST, k) * EAST +
                                        diffusion_coeff1(WEST, k) * WEST)
        elif coeff == 2:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 (diffusion_coeff2(NORTH, k) * NORTH +
                                        diffusion_coeff2(SOUTH, k) * SOUTH +
                                        diffusion_coeff2(EAST, k) * EAST +
                                        diffusion_coeff2(WEST, k) * WEST)
        else:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 (custom_coeff(NORTH, k) * NORTH +
                                        custom_coeff(SOUTH, k) * SOUTH +
                                        custom_coeff(EAST, k) * EAST +
                                        custom_coeff(WEST, k) * WEST)
        img = img_new

    return img

def diffusion(img, iters, k, coeff=1):
    #http://image.diku.dk/imagecanon/material/PeronaMalik1990.pdf
    img = img / 255
    img_new = np.zeros(img.shape, dtype=img.dtype)

    for step in range(iters):

        NORTH = img[:-2, 1:-1] - img[1:-1, 1:-1]

        SOUTH = img[2:, 1:-1] - img[1:-1, 1:-1]

        EAST = img[1:-1, 2:] - img[1:-1, 1:-1]

        WEST = img[1:-1, :-2] - img[1:-1, 1:-1]

        if coeff == 1:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 (diffusion_coeff1(NORTH, k) * NORTH +
                                        diffusion_coeff1(SOUTH, k) * SOUTH +
                                        diffusion_coeff1(EAST, k) * EAST +
                                        diffusion_coeff1(WEST, k) * WEST)
        elif coeff == 2:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 (diffusion_coeff2(NORTH, k) * NORTH +
                                        diffusion_coeff2(SOUTH, k) * SOUTH +
                                        diffusion_coeff2(EAST, k) * EAST +
                                        diffusion_coeff2(WEST, k) * WEST)
        else:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 (custom_coeff(NORTH, k) * NORTH +
                                        custom_coeff(SOUTH, k) * SOUTH +
                                        custom_coeff(EAST, k) * EAST +
                                        custom_coeff(WEST, k) * WEST)
        img = img_new

    return img