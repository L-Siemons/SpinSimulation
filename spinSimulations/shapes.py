import numpy as np
from scipy import integrate
import inspect

from typing import Callable, List, Union, Optional

# TODO: put it in the file?
# first line    -- center
# second line   -- amplitude
# third line    -- FWH
gauss_pulses_params = {
    'G3': {
        'center': np.array([0.287, 0.508, 0.795]),
        'ampl'  : np.array([-1, 1.37, 0.49]), 
        'width' : np.array([0.189, 0.183, 0.243])
    },
    'G4': {
        'center': np.array([0.177, 0.492, 0.653, 0.892]),
        'ampl'  : np.array([0.62, 0.72, -0.91, -0.33]), 
        'width' : np.array([0.172, 0.129, 0.119, 0.139])
    },
    'Q3': {
        'center': np.array([0.306, 0.545, 0.804]),
        'ampl'  : np.array([-4.39, 4.57, 2.6]), 
        'width' : np.array([0.180, 0.183, 0.245])
    },
    'Q5': {
        'center': np.array([0.162, 0.307, 0.497, 0.525, 0.803]),
        'ampl'  : np.array([-1.48, -4.34, 7.33, -2.30, 5.66]), 
        'width' : np.array([0.186, 0.139, 0.143, 0.290, 0.137])
    }
}

composite_pulse_params = {
    'CORPSE90': {
        'angle': np.array([384, 318.6, 24.3]),
        'phase': np.array([0, np.pi, 0])
    },
    'COMP180': {
        'angle': np.array([90, 180, 90]),
        'phase': np.array([0, 0.5 * np.pi, 0])    
    },
    'SCROFULOUS90': {
        'angle': np.array([115.2, 180, 115.2]),
        'phase': np.array([62.0, 280.6, 62.0]) * np.pi / 180        
    }
}


def combine_shapes(
    tau_p,
    shape_list: list, 
    time_list: Optional[Union[List[float], np.ndarray]]= None,
    scale_list: Optional[Union[List[float], np.ndarray]]= None,
    phase_list: Optional[Union[List[float], np.ndarray]]=None):
    
    if time_list is None:
        time_list = np.ones(shape=len(shape_list), dtype=np.float32) / len(shape_list)
    else:
        assert len(time_list) == len(shape_list), 'Shape list and time list should be equal'
        time_list = np.array(time_list)
        time_list = time_list / time_list.sum()
        assert np.all(time_list >= 0), 'Elements of time_list should be positive'
        
    if scale_list is None:
        scale_list = np.ones(shape=len(shape_list), dtype=np.float32)
    else:
        assert len(scale_list) == len(shape_list), 'Shape list and scale list should be equal'
        scale_list = np.array(scale_list) / max(np.abs(scale_list))
        
    if phase_list is None:
        phase_list = np.zeros(shape=len(shape_list), dtype=np.float32)
    else:
        assert len(phase_list) == len(shape_list), 'Shape list and scale list should be equal'
        
    time_cumsum = np.cumsum(time_list) * tau_p
    def _inner_func(ts):
        # Ensure ts is at least a 1D array
        ts = np.atleast_1d(ts)
        ampls = np.zeros_like(ts)
        phases = np.zeros_like(ts)
        
        cumulative_mask = np.zeros_like(ts, dtype=bool)
        for i, (shape, t_now, ph_now, scale_now) in enumerate(zip(shape_list, time_cumsum, phase_list, scale_list)):
            # Update phases only for elements not already updated
            mask = (ts < t_now) & ~cumulative_mask
            temp_ts = ts if i == 0 else ts - time_cumsum[i-1]
            ampl_now, phase_now = shape(tau_p * time_list[i])(temp_ts[mask])
            ampls[mask] = ampl_now * scale_now
            phases[mask] = (phase_now + ph_now)
            cumulative_mask |= mask  # Mark these elements as updated
            
        return (ampls[0], phases[0]) if ampls.size == 1 else (ampls, phases)
    return _inner_func
        
def gauss(
    tau_p, truncation=1, 
    sigma=None, mu=0.5, 
    percent=True, phase=0.):
    """
    IF SIGMA IS PROVIDED, IT OVERRIDES TRUNCATION
    """
    if sigma is not None:
        sigma = _sigma_from_trunc(truncation)
    else:
        if percent:
            truncation = truncation / 100
        sigma = _sigma_from_trunc(truncation)
    
    mu = mu 
    sigma = sigma 
    def _inner_func(ts):
        ts = np.atleast_1d(ts)
        xs = ts / tau_p - mu
        xs = xs * (1 - np.abs(0.5 - mu))
        ampls = np.exp(-0.5 * (xs / sigma)**2) 
        phases = np.ones(ts.shape) * phase
        return (ampls[0], phases[0]) if ampls.size == 1 else (ampls, phases)
    return _inner_func

def _sigma_from_trunc(truncation):
    return np.sqrt(1 / (8 * np.log(1 / truncation)))

def _trunc_from_sigma(tau_p, sigma):
    return np.exp(-(1/8) * (tau_p / sigma)**2)

def scale_90_to_gauss(
    truncation_factor=None, 
    sigma=None, mu=0.5, percent=True):
    
    shape = gauss(1, truncation_factor, sigma, mu, percent)
    
    return 1 / integrate.quad(shape, 0, 1)[0]

# === SINC === #
def sinc(tau_p, n:int=3, mu=0.5):
    # mu = tau_p * mu
    def _inner_func(ts):
        return np.sinc(2 * n * (ts / tau_p - mu))
    return _inner_func

def sinc_pp(tau_p, phase=0, mu=0.5, null=5, window=False):
    max_ampl = 0.5477225575051661
    def _inner_func(ts):
        ts = np.atleast_1d(ts)
        x_init = ts/tau_p - mu
        x_init = x_init * (1 - np.abs(0.5 - mu))
        x = x_init * 2 * null
        ampls = np.sinc(x)*np.sqrt(.3+x**2+x**4)*np.exp(-(x/2.3)**2) / max_ampl
        # ampls[abs(x_init) >= 0.5] = 0 # cut the sholders 
        if window:
            # It is hunning window
            ampls *= 0.5 * (1 - np.cos(2 * np.pi * x / 10))
        phases = np.full(ampls.shape, 0, dtype=np.float64)
        phases[ampls < 0] = np.pi
        phases += phase
        ampls = np.abs(ampls)
        # If the input was a scalar, return a scalar output
        return (ampls[0], phases[0]) if ampls.size == 1 else (ampls, phases)
    
    return _inner_func

def sinc_pp_90(tau_p, phase=0):
    max_ampl = 0.5477225575051661
    def _inner_func(ts):
        ts = np.atleast_1d(ts)
        x_init = ts/tau_p
        x_init = x_init * 0.4
        x = x_init * 10
        mask = (np.abs(x) <= 1)  # Speed up the central peak for small x
        x = np.where(mask, np.sign(x) * (2 * np.abs(x) - 1), x)
        ampls = np.sinc(x)*np.sqrt(.3+x**2+x**4)*np.exp(-(x/2.3)**2) / max_ampl
        mask = (x == 0)
        ampls = np.where(mask, 0, ampls)
        phases = np.full(ampls.shape, 0, dtype=np.float64)
        phases[ampls < 0] = np.pi
        phases += phase
        ampls = np.abs(ampls)
        # If the input was a scalar, return a scalar output
        return (ampls[0], phases[0]) if ampls.size == 1 else (ampls, phases)
    
    return _inner_func

def sinc_pp_2(tau_p, phase=0, mu=0.5):
    vals = np.array(
        [-0.10020513, -0.01518408, 0.00409132, -0.03173554, -0.03864283,
        -0.34812672, -0.01761114, 0.59418421, -2.12299696, 1.19193344]
    )
    max_ampl = 0.2313
    def _inner_func(ts):
        ts = np.atleast_1d(ts)
        x_init = ts/tau_p - mu
        x_init = x_init * (1 - np.abs(0.5 - mu))
        
        scale = 22
        x = x_init * scale
        x = x + vals[0]*(0.05*x)**2 + vals[1]*(0.05*x)**4
        ampls = np.sinc(x)
        for i, ampl in enumerate(vals[2:]):
            ampls += np.sinc(x) * ampl * np.cos(np.pi*x/(1+0.6*i))
        ampls /= max_ampl
        #  ampls[abs(x_init) >= 0.5] = 0 # cut the sholders 
        phases = np.full(ampls.shape, 0, dtype=np.float64)
        phases[ampls < 0] = np.pi
        phases += phase
        ampls = np.abs(ampls)
        # If the input was a scalar, return a scalar output
        return (ampls[0], phases[0]) if ampls.size == 1 else (ampls, phases)
    
    return _inner_func

# === G3, G4, Q3, Q5 === #
def ampl_pulse(tau_p, name='G3', phase=0., window=False):
    params = gauss_pulses_params
    
    if name not in params.keys():
        raise ValueError(f'Name of the shape {name} is not found')
    
    t_centers   = params[name]['center'][None, :] * tau_p       # where are the maximum 
                                                                # relative to overall time
    omega_maxs  = params[name]['ampl'][None, :]                 # maximal amplitude
    tau_halfs    = params[name]['width'][None, :] * 0.5 * tau_p # relative 
    omega_maxs = omega_maxs /  np.abs(omega_maxs).max()

    # recalculate sigmas using half-width
    inv_sigmas = np.log(2) / tau_halfs**2
    def _inner_func(ts):
        # ts.shape = (n_points, )
        ts = np.atleast_1d(ts)
        ampls = (
            (
                omega_maxs * np.exp(-inv_sigmas * (ts[:, None] - t_centers)**2)
            ).sum(axis=1)
        )
        if window:
            # It is hunning window
            ampls *= 0.5 * (1 - np.cos(2 * np.pi * ts / tau_p))
        phases = np.full(ampls.shape, 0, dtype=np.float64)
        phases[ampls < 0] = np.pi
        phases += phase
        ampls = np.abs(ampls)
        # If the input was a scalar, return a scalar output
        return (ampls[0], phases[0]) if ampls.size == 1 else (ampls, phases)
        
    return _inner_func

### 
def hanning(tau_p):
    def _inner_func(ts):
        # ts.shape = (n_points, )
        ts = np.atleast_1d(ts)
        result = 0.5 * (1 - np.cos(2 * np.pi * ts / tau_p))
        return result[0] if result.size == 1 else result
    return _inner_func

def square(tau_p, ph=0.):
    def _inner_func(ts):
        # ts.shape = (n_points, )
        ts = np.atleast_1d(ts)
        ampls = np.ones(ts.shape)
        phs = np.ones(ts.shape) * ph
        return (ampls[0], phs[0]) if ampls.size == 1 else (ampls, phs)
    return _inner_func


# TODO: should be able to provide the dict also
def comp_pulse(tau_p, phase=0, name='CORPSE90', shape=None):
    anlges = composite_pulse_params[name]['angle']
    time_fracs = anlges / anlges.sum()
    time_cumsum = np.cumsum(time_fracs) * tau_p
    phs = composite_pulse_params[name]['phase'] + phase
    def _inner_func(ts):
        # Ensure ts is at least a 1D array
        ts = np.atleast_1d(ts)
        if shape is None:
            ampls = np.ones_like(ts)
        else:
            # ampls, _ = shape(tau_p)(ts) # get rid of the phase
            ampls = np.zeros_like(ts)
        phases = np.zeros_like(ts)
        cumulative_mask = np.zeros_like(ts, dtype=bool)

        for i, time_current in enumerate(time_cumsum):
            # Update phases only for elements not already updated
            mask = (ts < time_current) & ~cumulative_mask
            phases[mask] = phs[i]
            cumulative_mask |= mask  # Mark these elements as updated
            if shape is not None:
                temp_ts = ts if i == 0 else ts - time_cumsum[i-1]
                current_ampl, _ = shape(tau_p * time_fracs[i])(temp_ts[mask])
                ampls[mask] = current_ampl
            
        return (ampls[0], phases[0]) if ampls.size == 1 else (ampls, phases)
    return _inner_func


# == SCALINGS === #
def scale_pulse_len(shape_func):
    
    # Validate that f is callable and takes exactly one argument
    if not callable(shape_func):
        raise TypeError("f must be a callable function.")
    
    # Inspect the function signature
    sig = inspect.signature(shape_func)
    if len(sig.parameters) != 1:
        raise ValueError("f must be a function that takes exactly one argument.")
    
    def fin_func(ts):
        f = shape_func(ts) 
        def _inner_func(x):
            ampl, ph = f(x)
            return ampl * np.cos(ph)
        return _inner_func 
    
    return 1 / integrate.quad(fin_func(1), 0, 1)[0]
    
    
# TODO: BLOCH CURRENTLY IN THE SHAPES
# BUT IT PROBABLY SHOULD BE IN THE SEPARATE FOLDER, CALLED BLOCH OR CLASSIC


## TODO: b1_func should a 3D vector, and everything should be rewrited as
## 3D
## Currently, B1 is along X axis
def bloch_rf(
    M, times, omega, b1_max=1, 
    pulse_func=lambda x : (1, 0),
    T1=None, T2=None, M0=1, is_hz=True):
    
    k1 = 0 if T1 is None else 1 / T1
    k2 = 0 if T2 is None else 1 / T2
    
    if is_hz:
        omega = omega * np.pi * 2
        b1_max = b1_max * np.pi * 2
    
    # def derive(M, t, b1_func):
        
    #     b1_loc_x = b1_max * b1_func(t) * np.cos(omega * t)
    #     b1_loc_y = b1_max * b1_func(t) * np.sin(omega * t)
        
    #     prop = np.array([
    #         [-k2,           0,              -b1_loc_y ],
    #         [0,             -k2,            b1_loc_x  ],
    #         [b1_loc_y,      -b1_loc_x,      -k1     ]
    #     ])
        
    #     end_state = np.array([0, 0, M0 * k1]) 
        
    #     return prop @ M + end_state
    
    def derive(M, t, pulse_func):
        
        b1, ph = pulse_func(t)
        # b1_loc_x = b1_max * b1_func(t) * np.cos(omega * t)
       #  b1_loc_y = b1_max * b1_func(t) * np.sin(omega * t)
        
        eps_x = b1_max * b1 * np.cos(ph)
        eps_y = b1_max * b1 * np.sin(ph)
         
        delta = omega
        
        prop = np.array([
            [-k2,           delta,               -eps_y],
            [-delta,        -k2,                eps_x ],
            [eps_y,           -eps_x,               -k1]
        ])
        
        end_state = np.array([0, 0, M0 * k1]) 
        
        return prop @ M + end_state
    
    return integrate.odeint(derive, M, times, args=(pulse_func,))

def bloch_profile(
    M_start, times, omegas, b1_max=1,
    pulse_func=lambda x : (1, 0), 
    is_hz=True):
    
    # Validate M_start
    if isinstance(M_start, str):
        if M_start not in ["x", "y", "z"]:
            raise ValueError("M_start must be a 3-element array summing to 1 or one of 'x', 'y', 'z'.")
        else:
            if M_start == 'x':
                M_start = np.array([1., 0., 0.], dtype=np.float32)
            elif M_start == 'y':
                M_start = np.array([0., 1., 0.], dtype=np.float32)
            elif M_start == 'z':
                M_start = np.array([0., 0., 1.], dtype=np.float32)
    elif isinstance(M_start, (list, np.ndarray)):
        M_start = np.array(M_start, dtype=float)
        if M_start.shape != (3,) or not np.isclose(np.sum(M_start), 1):
            raise ValueError("M_start must be a 3-element array summing to 1 or one of 'x', 'y', 'z'.")
    else:
        raise ValueError("M_start must be a 3-element array summing to 1 or one of 'x', 'y', 'z'.")
    
    mags = np.zeros(shape=(len(omegas), 3))
    for idx_nu, omega in enumerate(omegas):
        mags[idx_nu, :] = bloch_rf(
            M=M_start,
            times=times,
            omega=omega,
            b1_max=b1_max,
            pulse_func=pulse_func,
            is_hz=is_hz
        )[-1]
    return mags