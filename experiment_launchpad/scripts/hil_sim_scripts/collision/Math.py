from numba import njit
import numpy as np
@njit
def cuda_dot(x1,y1,z1,x2,y2,z2):
    return x1*x2+y1*y2+z1*z2

@njit
def cuda_vec3_minus(left,right):
    return left[0]-right[0],left[1]-right[1],left[2]-right[2]

@njit
def norm(x):
    '''
    norm(x) is a numba compiled 2-norm function for a (n,) numpy array
    
    @param x: (n,) np array
    @return float length
    
    on a (12,) vector it is approximately 10x as fast as np.linalg.norm
    '''
    return np.sqrt(np.sum(x*x))