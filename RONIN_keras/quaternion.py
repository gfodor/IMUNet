"""
Simple quaternion implementation for IMUNet compatibility
"""
import numpy as np

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        if isinstance(w, (list, tuple, np.ndarray)) and len(w) == 4:
            self.w, self.x, self.y, self.z = w
        else:
            self.w, self.x, self.y, self.z = w, x, y, z
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
    
    def conj(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def to_array(self):
        return np.array([self.w, self.x, self.y, self.z])

def quaternion(w, x=0.0, y=0.0, z=0.0):
    """Create a quaternion"""
    return Quaternion(w, x, y, z)

def from_float_array(arr):
    """Create quaternions from numpy arrays"""
    if arr.ndim == 1:
        if len(arr) == 4:
            return Quaternion(*arr)
        else:
            return Quaternion(0, arr[0], arr[1], arr[2] if len(arr) > 2 else 0)
    else:
        result = []
        for row in arr:
            if len(row) == 4:
                result.append(Quaternion(*row))
            else:
                result.append(Quaternion(0, row[0], row[1], row[2] if len(row) > 2 else 0))
        return result

def as_float_array(q_list):
    """Convert quaternions to numpy arrays"""
    if isinstance(q_list, Quaternion):
        return q_list.to_array()
    elif isinstance(q_list, (list, tuple)):
        return np.array([q.to_array() for q in q_list])
    else:
        # Assume it's already an array
        return np.array(q_list)