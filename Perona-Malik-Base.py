import numpy as np
def diffusion_coeff1(I, k):
    # exponential
    #np.exp(-1 * (np.power(lam, 2)) / (np.power(b, 2)))
    inside = (I**2)/(k**2)
    return np.exp(-inside)

def diffusion_coeff2(lam, b):
    return np.exp(-1 * (np.power(lam, 2)) / (np.power(b, 2)))


def anisodiff(im, steps, b, lam=0.25):  # takes image input, the number of iterations,

    im_new = np.zeros(im.shape, dtype=im.dtype)
    for t in range(steps):
        dn = im[:-2, 1:-1] - im[1:-1, 1:-1]
        ds = im[2:, 1:-1] - im[1:-1, 1:-1]
        de = im[1:-1, 2:] - im[1:-1, 1:-1]
        dw = im[1:-1, :-2] - im[1:-1, 1:-1]
        im_new[1:-1, 1:-1] = im[1:-1, 1:-1] + \
                             lam * (f(dn, b) * dn + f(ds, b) * ds +
                                    f(de, b) * de + f(dw, b) * dw)
        im = im_new
    return im