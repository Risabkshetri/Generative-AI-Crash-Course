import numpy as np

# vectors:

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

# dot product
dot_product = np.dot(v1,v2)
print(dot_product)  # Output: 32

scaler_multiplication = v1 * 2
print(scaler_multiplication) # Output: [2 4 6]

# matrices
matrix1 = np.array([[1,2],[3,4]])
matrix2 = np.array([[5,6],[7,8]])

# operations
matrix_product = matrix1 * matrix2 # element-wise multiplication
matrix_product = np.dot( matrix1, matrix2 ) # matrix multiplication
print(matrix_product)   # Output: [[19 22], [43 50]]

matrix_sum = np.add(matrix1, matrix2)
print(matrix_sum)  # Output: [[6 8], [10 12]]

matrix_transpose = np.transpose(matrix1)
print(matrix_transpose) # Output: [[1 3], [2 4]]

matrix_inv = np.linalg.inv(matrix2) 
print(matrix_inv) # Output: [[-0.0625  0.0625], [ 0.

matrix_eig = np.linalg.eig(matrix1)
print(matrix_eig) # Output: (array([ 3.37228132+0.j, -0. 
