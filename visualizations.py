import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
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

coeff_vs_grad()