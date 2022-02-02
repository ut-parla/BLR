import cupy as cp
import numpy as np
from timer import Timer
from common import cp_partition_array, partition_matrix

def gpu_direct(A, x, partition_size):
    Ad = cp.asarray(A)
    xd = cp.asarray(x)
    with Timer.get_handle("gpu_direct"):
        b1 = cp.matmul(Ad, xd)

def cp_TSVD(A, tol=1e-5):
	U,S,Vh = cp.linalg.svd(A)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ cp.diag(S[:k]), Vh[:k, :]


def gpu_BLR(A, x, partition_size):
    n_partitions = A.shape[1] // partition_size
    cp_UVs = {}

    A_partitions_rows = partition_matrix(A, partition_size)
    cp_A_partitions_rows = cp.asarray(A_partitions_rows)

    with Timer.get_handle("cupy-SVD"):
        for i in range(n_partitions):
            cp_UVs[i] = {}
            for j in range(n_partitions):
                cp_UVs[i][j] = cp_TSVD(cp_A_partitions_rows[i][j])

    cp_b2_rhs = cp.asarray(x)
    cp_b2_lhs = cp.zeros(x.shape)

    cp_b2_rhs_split = cp_partition_array(cp_b2_rhs, n_partitions)
    cp_b2_lhs_split = cp_partition_array(cp_b2_lhs, n_partitions)

    with Timer.get_handle("cupy-BLR-approx"):
        for i in range(n_partitions):
            for j in range(n_partitions):
                cp_b2_lhs_split[i] += cp.matmul(cp_UVs[i][j][0], cp.matmul(cp_UVs[i][j][1], cp_b2_rhs_split[j]))
