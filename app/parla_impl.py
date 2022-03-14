import cupy as cp
import numpy as np
from timer import Timer
from common import partition_array, partition_matrix

from parla import Parla
from parla.array import copy, clone_here
from parla.cuda import *
from parla.cpu import *
from parla.tasks import *
from parla.function_decorators import *

from parla.parray import asarray_batch, asarray

def cp_TSVD(A, tol=1e-5):
	U,S,Vh = cp.linalg.svd(A.array)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ cp.diag(S[:k]), Vh[:k, :]

def parla_BLR(A, x, partition_size):
     with Parla():
        @spawn()
        async def main():
            await __parla_BLR(A, x, partition_size)

async def __parla_BLR(A, x, partition_size):
    with Timer.get_handle("parla-setup"):
        n_partitions = A.shape[1] // partition_size
        
        A_partitions_rows = partition_matrix(A, partition_size)
        A_partitions = asarray_batch(A_partitions_rows)

        x_rhs = np.array(x, copy=True)
        x_rhs_split = partition_array(x_rhs, n_partitions)
        x_rhs_parray = asarray_batch(x_rhs_split)

        x_lhs = {}
        for i in range(len(x_rhs_split)):
            x_lhs[i] = []

    cp_UVs = {}
    svd_TS = TaskSpace("SVD")
    with Timer.get_handle("parla-SVD"):
        for i in range(n_partitions):
            cp_UVs[i] = {}
            for j in range(n_partitions):
                cp_UVs[i][j] = [asarray([]),asarray([])]
                @spawn(svd_TS[i, j], placement=gpu, input=[A_partitions], output=[*cp_UVs[i][j]])
                def tsvd_task():
                    u, v = cp_TSVD(A_partitions[i][j])
                    cp_UVs[i][j][0].update(u) 
                    cp_UVs[i][j][1].update(v)
        await svd_TS

    BLR_TS = TaskSpace("BLR")
    with Timer.get_handle("parla-BLR-approx"):
        b_lhs = {}
        #create a subarray to store results. if we use less we might serialize execution
        for i in range(n_partitions):
            b_lhs[i] = {}
            for j in range(n_partitions):
                b_lhs[i][j] = asarray(np.zeros_like(x_rhs_split[i]))

        for i in range(n_partitions):
            for j in range(n_partitions):
                #print(b_lhs[i][j])
                @spawn(BLR_TS[i, j], placement=gpu, input=[*cp_UVs[i][j], x_rhs_parray[i]], output=[b_lhs[i][j]])
                def blr_task():
                    #print(cp_UVs[i][j][1])
                    temp = cp.matmul(cp_UVs[i][j][1].array, x_rhs_parray[i].array)
                    b_lhs[i][j] = cp.matmul(cp_UVs[i][j][0].array, temp)
        await BLR_TS

    #could be done on cpu
    with Timer.get_handle("parla-accumulate"):
        final_lhs = np.zeros_like(x_rhs)
        final_lhs_split = partition_array(final_lhs, n_partitions)

        #accumulate sum of all i,j of b_lhs
        for i in range(n_partitions):
            for j in range(n_partitions):
                final_lhs_split[i] += b_lhs[i][j].get()

        res = np.concatenate(final_lhs_split, axis=None)
        #print("parla res: ", res)

