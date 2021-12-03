import sys
import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

M = 10
N = 10

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

def reduce(P1, P2):
    return P1 + P2

def concat_1d(P1, P2):
    return np.block([P1, P2])

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
PAx = concat_1d(reduce(P1x, P2x), reduce(P3x, P4x))
print(f"Multiplication of Ax by Partition: \n{PAx}")
print(f"Is multiplication correct? {np.allclose(PAx, Ax, rtol=0)}")

print(f"Decomposition of P1: \n{np.linalg.svd(P1)}")

def TSVD(A, tol=1e-5):
	U,S,Vh = np.linalg.svd(A)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ np.diag(S[:k]), Vh[:k, :]

U, V = TSVD(P1)
print(f"U,V of P1: \n{U}\n{V}")

P1U, P1V = TSVD(P1)
P1approx = P1U @ (P1V @ x1)

P2U, P2V = TSVD(P2)
P2approx = P2U @ (P2V @ x2)

P3U, P3V = TSVD(P3)
P3approx = P3U @ (P3V @ x1)

P4U, P4V = TSVD(P4)
P4approx = P4U @ (P4V @ x2)

Aapprox = concat_1d(reduce(P1approx, P2approx), reduce(P3approx, P4approx))
print(f"BLR of Ax: \n{PAx}")
print(f"Absolute diff of Ax and BLR: {Ax-Aapprox}")