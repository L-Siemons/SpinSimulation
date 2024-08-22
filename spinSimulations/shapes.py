import numpy as np
from scipy import integrate

def gauss(tau_p, truncation_factor=None, 
          sigma=None, mu=0.5, percent=True):
    if truncation_factor is None and sigma is None:
        raise Exception(
            'Either sigma or truncation should be provided'
        )

    if truncation_factor is not None and sigma is not None:
        raise Exception(
            'You cannot provide both sigma and '
            'truncation factor'
        )
    
    if truncation_factor is not None:
        if percent:
            truncation_factor = truncation_factor / 100
        sigma = _sigma_from_trunc(truncation_factor)
    
    mu = mu * tau_p
    sigma = sigma * tau_p
    def _inner_func(t):
        return np.exp(-0.5 * (t - mu)**2 / sigma**2)
    return _inner_func

def _sigma_from_trunc(truncation_factor):
    return np.sqrt(1 / (8 * np.log(1 / truncation_factor)))

def _trunc_from_sigma(tau_p, sigma):
    return np.exp(-(1/8) * (tau_p / sigma)**2)

def scale_90_to_gauss(
    truncation_factor=None, 
    sigma=None, mu=0.5, percent=True):
    
    shape = gauss(1, truncation_factor, sigma, mu, percent)
    
    return 1 / integrate.quad(shape, 0, 1)[0]