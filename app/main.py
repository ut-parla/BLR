import sys, os
import numpy as np
from timer import Timer
from cpu import cpu_direct, cpu_BLR, cpu_partition
from single_gpu import gpu_BLR, gpu_direct
from multi_gpu import mgpu_BLR
from parla_impl import parla_BLR

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

#M = 100000
#N = 100000
#partition_size = 10000

M = 100
N = 100
partition_size = 50


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

def main():
    A = random_matrix_svd()
    x = np.random.randn(N)

    #cpu_direct(A, x, partition_size)
    #cpu_partition(A, x, partition_size)
    #cpu_BLR(A, x, partition_size)

    #gpu_direct(A, x, partition_size)
    #gpu_BLR(A, x, partition_size)

    mgpu_BLR(A, x, partition_size)

    parla_BLR(A, x, partition_size)

    Timer.print()

    #os.kill(os.getpid(),11)


if __name__ == "__main__":
    main()



