import numpy as np

def diffusion_coeff1(I, k):
    # exponential
    #np.exp(-1 * (np.power(lam, 2)) / (np.power(b, 2)))
    inside = (I**2)/(k**2)
    return np.exp(-inside)

def diffusion_coeff2(I, k):
    denom = 1 + (np.power(I, 2)/np.power(k, 2))
    return 1/denom


def diffusion(img, iters, k, lam=0.12, coeff=1):

    img = img / 255
    img_new = np.zeros(img.shape, dtype=img.dtype)

    for step in range(iters):

        NORTH = img[:-2, 1:-1] - img[1:-1, 1:-1]

        SOUTH = img[2:, 1:-1] - img[1:-1, 1:-1]

        EAST = img[1:-1, 2:] - img[1:-1, 1:-1]

        WEST = img[1:-1, :-2] - img[1:-1, 1:-1]

        if coeff == 1:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 lam * (diffusion_coeff1(NORTH, k) * NORTH +
                                        diffusion_coeff1(SOUTH, k) * SOUTH +
                                        diffusion_coeff1(EAST, k) * EAST +
                                        diffusion_coeff1(WEST, k) * WEST)
        else:
            img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
                                 lam * (diffusion_coeff2(NORTH, k) * NORTH +
                                        diffusion_coeff2(SOUTH, k) * SOUTH +
                                        diffusion_coeff2(EAST, k) * EAST +
                                        diffusion_coeff2(WEST, k) * WEST)
        img = img_new

    return img