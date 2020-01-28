import numpy as np
"""
A = np.array([1,2,3,4])
print(A)

print(np.ndim(A)) # 배열의 차원수

print(A.shape) #배열의 형상

print(A.shape[0])
"""
"""
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(B)

print(np.ndim(B))

print(B.shape)

print(np.dot(A,B)) #행렬의 곱
"""
"""
A = np.array([[1,2,3], [4,5,6]])
B = np.array([[1,2], [3,4], [5,6]])

print(B)

print(A.shape)
print(B.shape)
print(np.dot(A,B))

C = np.array([[1,2], [3,4]])

print(C.shape)

print(np.dot(A,C))
"""
"""
A = np.array([[1,2], [3,4], [5,6]])
print(A.shape)

B = np.array([7,8])
print(B.shape)

print(np.dot(A,B))
"""
X = np.array([1,2])
print(X.shape)

W = np.array([[1,3,5], [2,4,6]])
print(W)
print(W.shape)
Y = np.dot(X, W)
print(Y)
