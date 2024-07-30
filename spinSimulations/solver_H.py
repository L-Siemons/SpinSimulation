### HERE SOLVERS FOR HAMILTONIAN FORMALISM WILL BE
import numpy as np
import scipy as sp
import scipy.sparse as sps
import functools
import types

def traj_prop_h(
        rho, 
        ham, 
        times, 
        traj_ops,
        system=None,
    ):
    if isinstance(ham, types.FunctionType):
        return _traj_prop_time_dependent(rho, ham, times, traj_ops,system=system)
    else: 
        return _traj_prop_time_independent(rho, ham, times, traj_ops,system=system)

def _set_ops(op, system=None):
    # matmul, elem_mul, kron, expm
    if system:
        return system.matmul, system.multiply, system.kron, system.expm
    
    if sps.issparse(op):
        return sparse_dot, sparse_multiply, sps.kron, sps.linalg.expm
    else:
        return np.matmul, np.multiply, np.kron, sp.linalg.expm
        
        

def prop_h(op, generator, time=None, system=None, reverse=False):
    
    matmul, _, _, expm = _set_ops(op, system=system)
    
    if time:
        prop = expm(-1j * generator * time)
    else:
        prop = expm(-1j * generator)
        
    if reverse:
        return matmul(
                prop.conj().T, matmul(op, prop)
            )
        
    return matmul(
            prop, matmul(op, prop.conj().T)
        )
         
        
def com_h(a, b, system=None):
    matmul, *_ = _set_ops(a, system=system)
    return matmul(a, b) - matmul(b, a)

def _traj_prop_time_dependent(
        rho=np.array([[]]), 
        ham=types.FunctionType, 
        times=np.array([]), 
        traj_ops=[],
        system=None,
    ):
    
    matmul, multiply, _ , expm = _set_ops(rho, system=system)
    
    ampls = [np.zeros_like(times, dtype=system.dtype) for _ in traj_ops]
    
        # evaulate timestep
    dt = times[1] - times[0]

    # propogate
    for idx, time in enumerate(times):
        prop = expm(-1j * ham(time) * dt)
        prop_dag = prop.conj().transpose()
        if idx != 0:
            rho = functools.reduce(
                matmul,
                [prop, rho, prop_dag]
            )
        for traj_op, ampl in zip(traj_ops, ampls):
            ampl[idx] = amplitude(traj_op, rho, multiply)
    return ampls

def _traj_prop_time_independent(
        rho=np.array([[]]), 
        ham=np.array([[]]), 
        times=np.array([]), 
        traj_ops=[],
        system=None,
    ):
    
    matmul, multiply, _ , expm = _set_ops(rho, system=system)
    
    ampls = [np.zeros_like(times, dtype=system.dtype) for _ in traj_ops]
    
        # evaulate timestep
    dt = times[1] - times[0]
    
    # define propogators
    prop = expm(-1j * ham * dt)
    prop_dag = prop.conj().transpose()
    
    # propogate
    for idx, _ in enumerate(times):
        if idx != 0:
            rho = functools.reduce(
                matmul,
                [prop, rho, prop_dag]
            )
        for traj_op, ampl in zip(traj_ops, ampls):
            ampl[idx] = amplitude(traj_op, rho, multiply)
    return ampls

def prop_uni(op, prop):
    return prop @ op @ prop.conj().T

def amplitude(op_to, op_from, multiply):
    """Calculate amplitude of operator_to in operator_from

    Args:
        op_to (_type_): _description_
        op_from (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return norm_frob(op_to, op_from, multiply=multiply) / norm_frob(op_to, multiply=multiply)

def norm_frob(op_first, op_second=None, multiply=None):
    """Calculate frobenious norm (squared)

    Args:
        op_first (_type_): _description_
        op_second (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    if op_second is None:
            return (multiply(op_first.conj(), op_first)).sum()
        
    return (multiply(op_first.conj(), op_second)).sum()


# TODO: it is a replica from a spin file, do something with it
def sparse_dot(a, b):
    """This is a helper function to provide two arguments multiplication function for sparse matrices

    Parameters
    ----------
    a : scipy sparse matrix
    b : scipy sparse matrix

    Returns
    -------
    scipy sparse matrix
    """
    return a.dot(b)

# TODO: the same as sparse dot
def sparse_multiply(a, b):
    return a.multiply(b)