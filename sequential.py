import sys
import numpy as np
from timer import Timer

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

M = 100
N = 100
partition_size = 20
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

A = random_matrix()
x = np.random.randn(N)

print(f"Shapes: A={A.shape} x={x.shape}")
print(f"# of partitions: {n_partitions}")

def partition_matrix(A, block_size):
    return A.reshape(A.shape[0] // block_size, block_size, A.shape[1] // block_size, block_size).swapaxes(1,2)

A_partitions_rows = partition_matrix(A, 20)
print(f"A_partitions_rows shape: {A_partitions_rows.shape}")

def partition_array(A, block_size):
    return np.split(A, block_size)

x_split = partition_array(x, n_partitions)
print(f"x_split shape: {len(x_split)} x {x_split[0].shape}")

#
# direct calculation
#
with Timer.get_handle("numpy-direct"):
    b1 = A @ x

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
with Timer.get_handle("SVD-partitions"):
    for i in range(n_partitions):
        UVs[i] = {}
        for j in range(n_partitions):
            UVs[i][j] = TSVD(A_partitions_rows[i][j])

print(f"Shape of UV for one partition's SVD decomposition {UVs[0][0][U].shape} {UVs[0][0][V].shape}")

#
# full approximation
#

b2 = x.copy()
b2_split = partition_array(b2, n_partitions)

with Timer.get_handle("full-svd-approximation"):
    for i in range(n_partitions):
        for j in range(n_partitions):
            #this is probably not correct
            b2_split[i] += UVs[i][j][U] @ (UVs[i][j][V] @ b2_split[i])

Timer.print()

print("Relative error:")
print((b2-b1)/b1)



sys.exit(0)
