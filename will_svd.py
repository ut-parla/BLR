import numpy as np
import time
N = 10
M = 10
A = np.random.randn(N, M)

def random_low_rank(A, bandwidth=0.7):
	U, S, V = np.linalg.svd(A)
	m = A.shape[1]
	idx = np.arange(m)
	S = S * np.exp(-idx/bandwidth) 
	return U @ np.diag(S) @ V

def TSVD(A, tol=1e-5):
	U,S,Vh = np.linalg.svd(A)
	scale = S/S[1]
	temp_k = np.argmax(scale < tol)
	k = temp_k if temp_k else A.shape[1]
	return U[:, :k] @ np.diag(S[:k]), Vh[:k, :]

B = random_low_rank(A)

tol = 1e-2
U, V = TSVD(B, tol)
print("Truncated Shape", U.shape, V.shape)
print("Memory Reduction", (B.shape[0]*B.shape[1])/(2*U.shape[0]*U.shape[1]), "x less memory")

x = np.random.rand(M)

t = time.perf_counter()
y1 = B @ x
t_full = time.perf_counter() - t

t = time.perf_counter()
y2 = U @ (V @ x)
t_comp = time.perf_counter() - t

print("Speedup", t_full/t_comp, "x faster")

print("Error in Output", np.linalg.norm(y1-y2))
print("Error in Matrix Approximation", np.linalg.norm(B - U @ V, ord='fro'))