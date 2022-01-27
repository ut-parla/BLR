import sys
import numpy as np
from timer import Timer

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

M = 1000
N = 1000
partition_size = 50
n_partitions = N // partition_size

#
# generate inputs
#

def random_matrix():
    return np.random.randn(M, N)

def random_matrix_svd(bandwidth=0.7):
    A = np.random.randn(M, N)
    U, S, V = np.linalg.svd(A)
    m = A.shape[1]
    idx = np.arange(m)
    S = S * np.exp(-idx/bandwidth) 
    return U @ np.diag(S) @ V

A = random_matrix_svd()
x = np.random.randn(N)

print(f"Shapes: A={A.shape} x={x.shape}")
print(f"# of partitions: {n_partitions}")

def partition_matrix(A, block_size):
    return A.reshape(A.shape[0] // block_size, block_size, A.shape[1] // block_size, block_size).swapaxes(1,2)

A_partitions_rows = partition_matrix(A, partition_size)
print(f"A_partitions_rows shape: {A_partitions_rows.shape}")

#print(A)
#print(A_partitions_rows)

def partition_array(A, block_size):
    return np.split(A, block_size)

def cp_partition_array(A, block_size):
    return cp.split(A, block_size)

x_split = partition_array(x, n_partitions)
print(f"x_split shape: {len(x_split)} x {x_split[0].shape}")

#
# direct calculation
#
with Timer.get_handle("numpy-direct"):
    b1 = A @ x

#
# by partition
#

b2_bypartition_rhs = x.copy()
b2_bypartition_lhs = np.zeros(x.shape)
b2_rhs_split = partition_array(b2_bypartition_rhs, n_partitions)
b2_lhs_split = partition_array(b2_bypartition_lhs, n_partitions)

with Timer.get_handle("by-partition"):
    for i in range(n_partitions):
        for j in range(n_partitions):
            b2_lhs_split[i] += (A_partitions_rows[i][j] @ b2_rhs_split[j])

print("Relative error of by partition and direct:")
print((b1-b2_bypartition_lhs)/b1)

#
# decompose
#
def TSVD(A, tol=1e-5):
	U,S,Vh = np.linalg.svd(A)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ np.diag(S[:k]), Vh[:k, :]

U = 0
V = 1

UVs = {}
with Timer.get_handle("decompose-SVD"):
    for i in range(n_partitions):
        UVs[i] = {}
        for j in range(n_partitions):
            UVs[i][j] = TSVD(A_partitions_rows[i][j])

print(f"Shape of UV for one partition's SVD decomposition {UVs[0][0][U].shape} {UVs[0][0][V].shape}")

#
# full approximation
#

b2_rhs = x.copy()
b2_lhs = np.zeros(x.shape)

b2_rhs_split = partition_array(b2_rhs, n_partitions)
b2_lhs_split = partition_array(b2_lhs, n_partitions)

with Timer.get_handle("SVD-mult-approximation"):
    for i in range(n_partitions):
        for j in range(n_partitions):
            b2_lhs_split[i] += UVs[i][j][U] @ (UVs[i][j][V] @ b2_rhs_split[j])

#
# SVD using cupy
#
import cupy as cp

def cp_TSVD(A, tol=1e-5):
	U,S,Vh = cp.linalg.svd(A)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ cp.diag(S[:k]), Vh[:k, :]

cp_UVs = {}

cp_A_partitions_rows = cp.asarray(A_partitions_rows)

with Timer.get_handle("cupy-decompose-SVD"):
    for i in range(n_partitions):
        cp_UVs[i] = {}
        for j in range(n_partitions):
            cp_UVs[i][j] = cp_TSVD(cp_A_partitions_rows[i][j])

cp_b2_rhs = cp.asarray(x)
cp_b2_lhs = cp.zeros(x.shape)

cp_b2_rhs_split = cp_partition_array(cp_b2_rhs, n_partitions)
cp_b2_lhs_split = cp_partition_array(cp_b2_lhs, n_partitions)

with Timer.get_handle("cupy-SVD-mult-approximation"):
    for i in range(n_partitions):
        for j in range(n_partitions):
            cp_b2_lhs_split[i] += cp.matmul(UVs[i][j][U], cp.matmul(UVs[i][j][V], b2_rhs_split[j]))




print("Relative error of direct and approx:")
print((b2_lhs-b1)/b1)

print("Relative error of by partition and approx:")
print((b2_lhs-b2_bypartition_lhs)/b1)

Timer.print()
sys.exit(0)
