import sys, os
import numpy as np
from timer import Timer
from cpu import cpu_direct, cpu_BLR, cpu_partition
from single_gpu import gpu_BLR, gpu_direct
from multi_gpu import mgpu_BLR
from parla_impl import parla_BLR

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

#
# generate inputs
#

#def random_matrix():
#    return np.random.randn(M, N)

# def random_matrix_svd(bandwidth=0.7):
#     A = np.random.randn(M, N)
#     U, S, V = np.linalg.svd(A)
#     m = A.shape[1]
#     idx = np.arange(m)
#     S = S * np.exp(-idx/bandwidth) 
#     return U @ np.diag(S) @ V

def random_matrix_svd(M, rank=10):
    A = np.random.randn(M, M)
    U, S, V = np.linalg.svd(A)
    m = A.shape[1]
    idx = np.arange(m)
    S[rank:] = 0
    return U @ np.diag(S) @ V

def main():
    if len(sys.argv) != 5:
        print("Need 3 arguments:")
        print("   : <run> <matrix file path> <array file path> <partition size>")
        print(" or: <gen> <size> <matrix file path> <array file path>")
        sys.exit(1)

    rtype = sys.argv[1]
    partition_size = 0
    if rtype == "gen":
        N = int(sys.argv[2])
        A = random_matrix_svd(N)
        np.save(sys.argv[3], A)
        x = np.random.randn(N)
        np.save(sys.argv[4], x)
        sys.exit(0)
    elif rtype == "run":
        A = np.load(sys.argv[2])
        x = np.load(sys.argv[3])
        partition_size = int(sys.argv[4])
        
    #cpu_direct(A, x, partition_size)
    #cpu_partition(A, x, partition_size)
    #cpu_BLR(A, x, partition_size)

    #gpu_direct(A, x, partition_size)
    #gpu_BLR(A, x, partition_size)

    #mgpu_BLR(A, x, partition_size)

    parla_BLR(A, x, partition_size)

    Timer.print()
    #os.kill(os.getpid(),11)

if __name__ == "__main__":
    main()



