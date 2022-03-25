import cupy as cp
import numpy as np
from timer import Timer
from common import cp_partition_array, partition_matrix, partition_array
import threading, queue
import logging
from functools import wraps
from itertools import product

class GPUCell:
    def __init__(self, resident_gpu):
        self.resident_gpu = resident_gpu
        self.queue = queue.Queue()
        self.done_event = threading.Event()
        self.thread = None
        self.keep_running = True
        self.threads_per_block = 256

    def launch(self):
        """Launch ourselves as thread"""
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        """Loop listening to the queue. Incoming messages from the 
        queue have function name to execute and optional args.
        Everything is run on the threads' GPU.
        """
        with cp.cuda.Device(self.resident_gpu):
            print(f"Thread running in cuda context, GPU #{self.resident_gpu}")
            while self.keep_running:
                # block until we get a fn to execute
                msg = self.queue.get()
                fn_name = msg[0]
                # if len is 1, no args
                if len(msg) == 1:
                    getattr(self, fn_name)()
                # else, expand args
                else:
                    args = msg[1]
                    getattr(self, fn_name)(*args)
        print(f"GPU {self.resident_gpu}: thread stopping...")

    def notify_when_done(fn):     
        @wraps(fn)
        def wrapper(*args, **kwargs):
            fn(*args, **kwargs)
            args[0].done_event.set()
        return wrapper    

    @notify_when_done
    def terminate(self):
        self.keep_running = False
            
    #A is a matrix of partitions, my_partitions is a list of (x,y)
    def set_TSVD_partitions(self, h_A_partitions, my_partitions):
        self.h_A_partitions = h_A_partitions
        self.my_partitions = my_partitions
        #print(f"partitions of GPU {self.resident_gpu}: {my_partitions}")

    @notify_when_done
    def set_X(self, h_x_partitions):
        self.d_lhs_X = {}
        self.d_rhs_X = {}

        for i, p in enumerate(h_x_partitions):
            self.d_lhs_X[i] = cp.zeros_like(p)
            self.d_rhs_X[i] = cp.array(p, copy=True)

    @notify_when_done
    def do_TSVD(self):
        self.d_UVs = {}

        for x,y in self.my_partitions:
            #print(f"GPU {self.resident_gpu} doing TSVD of {x}/{y}")
            if x not in self.d_UVs.keys(): self.d_UVs[x] = {}
            part = cp.array(self.h_A_partitions[x][y], copy=True)
            self.d_UVs[x][y] = cp_TSVD(part)

    @notify_when_done
    def do_BLR(self):
        for x,y in self.my_partitions:
            self.d_lhs_X[x] += cp.matmul(self.d_UVs[x][y][0], cp.matmul(self.d_UVs[x][y][1], self.d_rhs_X[x]))

gpu_cells_initd = False
ngpus = 4
gpu_cells = {}
def init_gpu_cells():
    global gpu_cells_initd
    #create the threads if we haven't
    if not gpu_cells_initd:
        for i in range(ngpus):
            gpu_cells[i] = GPUCell(i)
            gpu_cells[i].launch()
        gpu_cells_initd = True

def call_method_all_cells(fn_name, *args):
    #send message to all threads
    for cell in gpu_cells.values():
        cell.queue.put((fn_name, args))

    for cell in gpu_cells.values():
        cell.done_event.wait()
        cell.done_event.clear()

def cp_TSVD(A, tol=1e-5):
	U,S,Vh = cp.linalg.svd(A)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ cp.diag(S[:k]), Vh[:k, :]

def mgpu_BLR(A, x, partition_size):
    with Timer.get_handle("multigpu-setup"):
        n_partitions = A.shape[1] // partition_size
        A_partitions_rows = partition_matrix(A, partition_size)
        x_split = cp.asarray(partition_array(x, n_partitions))
        partitions_list = list(product(range(n_partitions), range(n_partitions)))
        partition_split = np.array_split(partitions_list, ngpus)
        init_gpu_cells()
        call_method_all_cells("set_X", x_split)

    with Timer.get_handle("multigpu-SVD"):
        for i, c in gpu_cells.items():
            c.set_TSVD_partitions(A_partitions_rows, partition_split[i])
        call_method_all_cells("do_TSVD")

    with Timer.get_handle("multigpu-BLR-approx"):
        call_method_all_cells("do_BLR")

    #could be done on gpu
    with Timer.get_handle("multigpu-accumulate"):
        final_lhs = np.zeros_like(x)
        final_lhs_split = partition_array(final_lhs, n_partitions)

        for worker in gpu_cells.values():
            for i, val in worker.d_lhs_X.items():
                final_lhs_split[i] += val.get()

        res = np.concatenate(final_lhs_split, axis=None)
        print("mgpu res: ", res)

    call_method_all_cells("terminate")