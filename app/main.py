import sys, os
import numpy as np
from timer import Timer

import argparse
parser = argparse.ArgumentParser()


#Which runtime to use?
parser.add_argument('-mode', type=str, default="run", help="Options: run, gen")
#Which runtime to use?
parser.add_argument('-type', type=str, default="parla", help="Options: parla, cpu_dot, cpu_BLR, mgpu_BLR, gpu_BLR, gpu_dot")
#Blocksize
parser.add_argument('-b', type=int, default=2000)
#How many blocks
parser.add_argument('-nblocks', type=int, default=4)
#How many trials to run
parser.add_argument('-trials', type=int, default=2)
#What matrix file (.npy) to load
parser.add_argument('-matrix', default="inputs/matrix.npy")
#What vector file (.npy) to load
parser.add_argument('-vector', default="inputs/vector.npy")
#Is the data movement automatic or manual?
parser.add_argument('-movement', default="lazy", type=str, help="Options: lazy, eager")
#Are the placements fixed by the user or determined by the scheduler?
parser.add_argument('-fixed', default=0, type=int)
#How many GPUs to run on?
parser.add_argument('-ngpus', default=4, type=int)
args = parser.parse_args()

cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:args.ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


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
    S[rank:] = 0
    return U @ np.diag(S) @ V

def make_BLR(M, source_matrices=None, block_size=1000):
    import random

    A = np.empty([M, M], dtype=np.float64)

    assert(M%block_size == 0)
    n_blocks = M//block_size

    if source_matrices is None:
        source_matrices = n_blocks*n_blocks

    matrix_list = []
    rank_list = [10]
    for i in range(source_matrices):
        k = random.choice(rank_list)
        matrix = random_matrix_svd(block_size, rank=k)
        matrix_list.append(matrix)

    for i in range(n_blocks):
        for j in range(n_blocks):
            matrix = random.choice(matrix_list)
            A[(i)*block_size:(i+1)*block_size, (j)*block_size:(j+1)*block_size] = matrix

    return A


def main():
    rtype = args.mode
    partition_size = 0
    if rtype == "gen":
        n_blocks = args.nblocks
        b = args.b
        N = n_blocks*b
        print("Generating input matrix", N, b)
        A = make_BLR(N, source_matrices=2, block_size=b)
        np.save(args.matrix, A)
        x = np.random.randn(N)
        np.save(args.vector, x)
        sys.exit(0)

    elif rtype == "run":
        A = np.load(args.matrix)
        x = np.load(args.vector)
        A = A.astype(np.float64)
        x = x.astype(np.float64)
        partition_size = int(args.b)
        ptype = args.type

        for _ in range(args.trials):
            if ptype == "cpudot":
                from cpu import cpu_direct
                cpu_direct(A, x, partition_size)
            elif ptype == "cpupart":
                from cpu import cpu_partition
                cpu_partition(A, x, partition_size)
            elif ptype == "cpu_blr":
                from cpu import cpu_BLR
                cpu_BLR(A, x, partition_size)
            elif ptype == "gpudot":
                from single_gpu import gpu_direct
                gpu_direct(A, x, partition_size)
            elif ptype == "gpu_blr":
                from single_gpu import gpu_BLR
                gpu_BLR(A, x, partition_size)
            elif ptype == "mgpu_blr":
                from multi_gpu import mgpu_BLR
                mgpu_BLR(A, x, partition_size)
            elif ptype == "parla":
                from parla_impl import parla_BLR
                manual_placement = True if args.fixed else False
                use_lazy = True if args.movement == "lazy" else False
                print(f"placement manual? {manual_placement}.")
                print(f"lazy alloc? {use_lazy}")
                parla_BLR(A, x, partition_size, manual_placement, use_lazy)

        #nwarm < nruns
        Timer.print(nwarm=1)

if __name__ == "__main__":
    main()

