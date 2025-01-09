import numpy as np

# vectors:

v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

# dot product
dot_product = np.dot(v1,v2)
print("\n--------Dot Product------\n",dot_product) 

cross_product = np.cross(v1, v2)
print("\n--------Cross Product------\n",cross_product)

scaler_multiplication = v1 * 2
print("\n--------Scaler Multiplication------\n",scaler_multiplication)

# matrices
matrix1 = np.array([[1,2],[3,4]])
matrix2 = np.array([[5,6],[7,8]])

# operations
matrix_product = matrix1 * matrix2 # element-wise multiplication
matrix_product = np.dot( matrix1, matrix2 ) # matrix multiplication
print("\n--------product------\n",matrix_product)  

matrix_sum = np.add(matrix1, matrix2)
print("\n--------sum------\n",matrix_sum)  

# matrix_transpose = np.transpose(matrix1)
matrix_transpose = matrix1.T
print("\n--------Transpose------\n", matrix_transpose) 

matrix_inv = np.linalg.inv(matrix2) 
print("\n--------Inverse------\n",matrix_inv)

matrix_eig = np.linalg.eig(matrix1)
print("\n--------Eigen value------\n",matrix_eig) 


# Linear Trasformation
# 1.Rotation Matrix
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

point = np.array([1,1])
theta = np.pi/4
rotated_point = rotation_matrix(theta) @ point
print("\n--------Rotated Point------\n",rotated_point)

# 2. Scaling Matrix

def scaling_matrix(sx, sy):
    return np.array([[sx, 0], [0, sy]])

point = np.array([1,1])
scaled_point = scaling_matrix(2, 3) @ point
print("\n--------Scaled Point------\n",scaled_point)