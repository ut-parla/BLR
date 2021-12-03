import sys
import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

M = 10
N = 10

A = np.random.randn(M, N)
x = np.random.randn(N)

print(f"Shapes: A={A.shape} x={x.shape}")
print(f"A: \n{A}")

P1 = A[0:5,0:5]
P2 = A[0:5,5:10]
P3 = A[5:10,0:5]
P4 = A[5:10,5:10]
x1, x2 = x[0:5], x[5:10]

print(f"Partitioning A into 4 blocks..")
print(f"Partition P1 \n{P1}")
print(f"Partition P2 \n{P2}")
print(f"Partition P3 \n{P3}")
print(f"Partition P4 \n{P4}")

def concat_1d(P1, P2, P3, P4):
    return np.block([P1 + P2, P3 + P4])

def concat_matrix(P1, P2, P3, P4):
    return np.block([[P1, P2], [P3, P4]])

print(f"Concatenation of partitions of A: \n{concat_matrix(P1, P2, P3, P4)}")
print(f"Is concatenation correct? {(concat_matrix(P1, P2, P3, P4) == A).all()}")

Ax = A@x
print(f"Multiplication of Ax: \n{Ax}")

P1x = P1@x1
P2x = P2@x2
P3x = P3@x1
P4x = P4@x2
PAx = concat_1d(P1x, P2x, P3x, P4x)
print(f"Multiplication of Ax by Partition: \n{PAx}")
print(f"Is multiplication correct? {(PAx == Ax).all()}")
print(f"Abs error: {PAx-Ax}")


P1u, P1s, P1vh = np.linalg.svd(P1)
print(f"Decomposition of P1: \n{P1u, P1s, P1vh}")



