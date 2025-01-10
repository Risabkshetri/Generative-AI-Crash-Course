from sympy import symbols, diff

x = symbols('x')

function = x**3 + 2*x**2 + 3*x + 4
diff = diff(function, x)
print(diff)

import numpy as np
import matplotlib.pyplot as plt

# Define a quadratic loss function
def loss_function(x):
    return x**2

# Derivative (gradient)
def gradient(x):
    return 2 * x

# Gradient descent implementation
x = 10  # Initial guess
learning_rate = 0.1
steps = []

for i in range(20):  # 20 iterations
    steps.append(x)
    x = x - learning_rate * gradient(x)

# Plotting the loss function and descent
x_vals = np.linspace(-10, 10, 100)
y_vals = loss_function(x_vals)

plt.plot(x_vals, y_vals, label="Loss Function")
plt.scatter(steps, [loss_function(s) for s in steps], color='red', label="Gradient Descent Steps")
plt.legend()
plt.title("Gradient Descent Visualization")
plt.xlabel("x")
plt.ylabel("Loss")
plt.show()
