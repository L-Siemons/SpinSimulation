### HERE SOLVERS FOR HAMILTONIAN FORMALISM WILL BE
import numpy as np
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

def _traj_prop_time_dependent(
        rho=np.array([[]]), 
        ham=types.FunctionType, 
        times=np.array([]), 
        traj_ops=[],
        system=None,
    ):
    
    ampls = [np.zeros_like(times, dtype=system.dtype) for _ in traj_ops]
    
        # evaulate timestep
    dt = times[1] - times[0]

    # propogate
    for idx, time in enumerate(times):
        prop = system.expm(-1j * ham(time) * dt)
        prop_dag = prop.conj().transpose()
        if idx != 0:
            rho = functools.reduce(
                system.matmul,
                [prop, rho, prop_dag]
            )
        for traj_op, ampl in zip(traj_ops, ampls):
            ampl[idx] = amplitude(traj_op, rho)
    return ampls

def _traj_prop_time_independent(
        rho=np.array([[]]), 
        ham=np.array([[]]), 
        times=np.array([]), 
        traj_ops=[],
        system=None,
    ):
    
    ampls = [np.zeros_like(times, dtype=system.dtype) for _ in traj_ops]
    
        # evaulate timestep
    dt = times[1] - times[0]
    
    # define propogators
    prop = system.expm(-1j * ham * dt)
    prop_dag = prop.conj().transpose()
    
    # propogate
    for idx, _ in enumerate(times):
        if idx != 0:
            rho = functools.reduce(
                system.matmul,
                [prop, rho, prop_dag]
            )
        for traj_op, ampl in zip(traj_ops, ampls):
            ampl[idx] = amplitude(traj_op, rho)
    return ampls

def amplitude(op_to, op_from):
    """Calculate amplitude of operator_to in operator_from

    Args:
        op_to (_type_): _description_
        op_from (_type_): _description_

    Returns:
        _type_: _description_
    """    
    return norm_frob(op_to, op_from) / norm_frob(op_to)

def norm_frob(op_first, op_second=None):
    """Calculate frobenious norm (squared)

    Args:
        op_first (_type_): _description_
        op_second (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    if op_second is None:
            return (op_first.conj() * op_first).sum()
        
    return (op_first.conj() * op_second).sum()