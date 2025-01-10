import numpy as np
import matplotlib.pyplot as plt

# statistics

data = [10,20,30,40,60]

median = np.median(data)
print("Median = ", median)

mean = np.mean(data)
print("Mean = ", mean)

var = np.var(data)
print("Variance = ", var)

std = np.std(data)
print("Standard Deviatioon = ", std)

# probability
# 1. Normal Distribution
mean = 0
std = 1
size = 1000

data = np.random.normal(mean, std, size)

plt.hist(data, bins=50, alpha=0.7, color='blue', label='Normal Distribution')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# 2. Binomial Distribution
n = 10
p = 0.5
size = 1000

data = np.random.binomial(n, p, size)
plt.hist(data, bins=10, alpha=0.7, color='blue', label='Binomial Distribution')
plt.title('Binomial Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 3. Poisson Distribution
lam = 2
size = 1000

data = np.random.poisson(lam, size)

plt.hist(data, bins=10, alpha=0.7, color='blue', label='Poisson Distribution')
plt.title('Poisson Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()