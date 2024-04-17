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

def first_deriv_x(I, i, j, delta=1):
    new = (I[i+delta, j] - I[i-delta, j])/(2*delta)
    return new
def first_deriv_y(I, i, j, delta=1):
    new = (I[i, j+delta] - I[i, j-delta])/(2*delta)
    return new

def second_deriv_x(I, i, j, delta=1):
    num = I[i+delta, j] - 2*I[i,j] + I[i-1,j]
    new = num / (delta**2)
    return new

def second_deriv_y(I, i, j, delta=1):
    num = I[i, j+delta] - 2*I[i,j] + I[i,j-delta]
    new = num / (delta**2)
    return new


def diffusion_(img, iters, k=0.1, lam=0.12, coeff=1):

    img = img / 255
    img_new = np.zeros(img.shape, dtype=img.dtype)

    for j in range(iters):

        for i in range(1, iters-1):
            Ix = first_deriv_x(img, i=i, j=j)
            Iy = first_deriv_y(img, i=i, j=j)
            Ixx = second_deriv_x(img, i=i, j=j)
            Iyy = second_deriv_y(img, i=i, j=j)
            exp_coef = (0.6/k**2)*np.exp((-3/(2*(k**2)))*(Ix**2+Iy**2))
            a = 3/(k**2)
            b = 6/(k**2)
            inside = Ixx - a*(Ix**2)*Ixx - b*Ix*Iy*Ixy + Iyy - a*(Iy**2)*Iyy
            img_new[i, j] += lam*(c)

        # NORTH = img[:-2, 1:-1] - img[1:-1, 1:-1]
        #
        # SOUTH = img[2:, 1:-1] - img[1:-1, 1:-1]
        #
        # EAST = img[1:-1, 2:] - img[1:-1, 1:-1]
        #
        # WEST = img[1:-1, :-2] - img[1:-1, 1:-1]
        #
        # if coeff == 1:
        #     img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
        #                          (diffusion_coeff1(NORTH, k) * NORTH +
        #                                 diffusion_coeff1(SOUTH, k) * SOUTH +
        #                                 diffusion_coeff1(EAST, k) * EAST +
        #                                 diffusion_coeff1(WEST, k) * WEST)
        # elif coeff == 2:
        #     img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
        #                          (diffusion_coeff2(NORTH, k) * NORTH +
        #                                 diffusion_coeff2(SOUTH, k) * SOUTH +
        #                                 diffusion_coeff2(EAST, k) * EAST +
        #                                 diffusion_coeff2(WEST, k) * WEST)
        # else:
        #     img_new[1:-1, 1:-1] = img[1:-1, 1:-1] + \
        #                          (custom_coeff(NORTH, k) * NORTH +
        #                                 custom_coeff(SOUTH, k) * SOUTH +
        #                                 custom_coeff(EAST, k) * EAST +
        #                                 custom_coeff(WEST, k) * WEST)
        # img = img_new

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