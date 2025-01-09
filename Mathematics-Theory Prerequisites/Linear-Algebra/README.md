# Linear Algebra Essentials

## Table of Contents
- [1. Vectors](#1-vectors)
- [2. Matrices](#2-matrices)
- [3. Linear Transformations](#3-linear-transformations)

## 1. Vectors
### Dot Product (Scalar Product)
```python
import numpy as np

# Dot product example
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Using NumPy's dot product
dot_product = np.dot(v1, v2)

# Manual calculation
manual_dot = sum(x*y for x, y in zip(v1, v2))

print(f"Dot product: {dot_product}")
```

### Cross Product (Vector Product)
```python
# Cross product (only for 3D vectors)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

cross_product = np.cross(v1, v2)
print(f"Cross product: {cross_product}")
```

## 2. Matrices

### Matrix Operations

#### Multiplication
```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Using NumPy's matrix multiplication
product = np.matmul(A, B)
# Alternative: product = A @ B

print("Matrix multiplication:")
print(product)
```

#### Sum and Transpose
```python
# Matrix addition
sum_matrix = A + B

# Matrix transpose
transpose_A = A.T

print("Matrix sum:")
print(sum_matrix)
print("\nTranspose:")
print(transpose_A)
```

#### Inverse
```python
# Matrix inverse
inverse_A = np.linalg.inv(A)
print("Inverse matrix:")
print(inverse_A)
```

#### Eigenvalues and Eigenvectors
```python
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)
```

For detailed visualization of eigenvalues and eigenvectors, check out this [video tutorial](https://youtu.be/PFDu9oVAE-g).

## 3. Linear Transformations

### Implementation Examples

#### Rotation Matrix (2D)
```python
def rotation_matrix(theta):
    """Create 2D rotation matrix for angle theta (in radians)"""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

# Example: 45-degree rotation
point = np.array([1, 0])
theta = np.pi/4  # 45 degrees
rotated_point = rotation_matrix(theta) @ point
```

#### Scaling Matrix
```python
def scaling_matrix(sx, sy):
    """Create 2D scaling matrix"""
    return np.array([[sx, 0], [0, sy]])

# Example: Scale by 2 in x and 3 in y
point = np.array([1, 1])
scaled_point = scaling_matrix(2, 3) @ point
```

For more detailed explanations of transformations, visit [Cuemath's transformation guide](https://www.cuemath.com/geometry/transformations/).


## Practice Exercises
1. Implement vector operations without using NumPy
2. Create a function to perform matrix multiplication from scratch
3. Implement a linear transformation that combines rotation and scaling

## Additional Resources
- [Interactive Linear Algebra Visualizations](https://github.com/3b1b/manim)
- [MIT OpenCourseWare: Linear Algebra](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

[← Back to Mathematics](README.md) | [← Back to Main](../../README.md)np
