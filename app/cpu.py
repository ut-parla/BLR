import sys
import numpy as np
from timer import Timer
from common import partition_matrix, partition_array


def cpu_direct(A, x, partition_size):
    with Timer.get_handle("cpu_direct"):
        b1 = A @ x


def cpu_partition(A, x, partition_size):
    n_partitions = A.shape[1] // partition_size

    A_partitions_rows = partition_matrix(A, partition_size)
    x_split = partition_array(x, n_partitions)

    b2_bypartition_rhs = x.copy()
    b2_bypartition_lhs = np.zeros(x.shape)
    b2_rhs_split = partition_array(b2_bypartition_rhs, n_partitions)
    b2_lhs_split = partition_array(b2_bypartition_lhs, n_partitions)

    with Timer.get_handle("cpu_partition"):
        for i in range(n_partitions):
            for j in range(n_partitions):
                b2_lhs_split[i] += (A_partitions_rows[i][j] @ b2_rhs_split[j])

#
# SVD part
#

def TSVD(A, tol=1e-5):
	U,S,Vh = np.linalg.svd(A)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ np.diag(S[:k]), Vh[:k, :]

UVs = {}

def cpu_BLR(A, x, partition_size):
    n_partitions = A.shape[1] // partition_size

    A_partitions_rows = partition_matrix(A, partition_size)

    with Timer.get_handle("cpu-SVD"):
        for i in range(n_partitions):
            UVs[i] = {}
            for j in range(n_partitions):
                UVs[i][j] = TSVD(A_partitions_rows[i][j])

    b2_rhs = x.copy()
    b2_lhs = np.zeros(x.shape)

    b2_rhs_split = partition_array(b2_rhs, n_partitions)
    b2_lhs_split = partition_array(b2_lhs, n_partitions)

    with Timer.get_handle("cpu-BLR"):
        for i in range(n_partitions):
            for j in range(n_partitions):
                b2_lhs_split[i] += UVs[i][j][0] @ (UVs[i][j][1] @ b2_rhs_split[j])