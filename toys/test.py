import sys
import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

M = 10
N = 10
block_size = 5
A = np.random.randn(M, N)
Ap = A.reshape(M // block_size, block_size, N // block_size, block_size).swapaxes(1,2)

print(A)
print(Ap)