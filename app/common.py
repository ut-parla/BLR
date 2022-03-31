import numpy as np
import cupy as cp

def partition_array(A, block_size):
    return np.split(A, block_size)

def cp_partition_array(A, block_size):
    return cp.split(A, block_size)

def partition_matrix(A, block_size):
    return A.reshape(A.shape[0] // block_size, block_size, A.shape[1] // block_size, block_size).swapaxes(1,2)

def actual_partition_matrix(A, block_size):
    ap_list = []
    n = A.shape[1]
    for i in range(n//block_size):
        ap_list.append(list())
        for j in range(n//block_size):
            ap_list[i].append(np.asarray(A[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size], order='F'))
    return ap_list