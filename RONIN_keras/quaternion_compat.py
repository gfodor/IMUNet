"""
Quaternion compatibility shim
This provides a quaternion interface compatible with the code's expectations
"""
import numpy as np
from pyquaternion import Quaternion as PyQuaternion

def quaternion(*args):
    """Create a quaternion from components"""
    if len(args) == 4:
        return PyQuaternion(w=args[0], x=args[1], y=args[2], z=args[3])
    elif len(args) == 1:
        return PyQuaternion(args[0])
    else:
        raise ValueError("Invalid quaternion arguments")

def from_float_array(arr):
    """Create quaternions from float arrays"""
    if arr.ndim == 1:
        return quaternion(*arr)
    else:
        return [quaternion(*row) for row in arr]

def as_float_array(q_list):
    """Convert quaternions to float arrays"""
    if isinstance(q_list, (list, tuple)):
        return np.array([[q.w, q.x, q.y, q.z] for q in q_list])
    else:
        return np.array([q_list.w, q_list.x, q_list.y, q_list.z])