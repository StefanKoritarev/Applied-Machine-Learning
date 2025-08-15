import numpy as np
import matplotlib.pyplot as plt


matrixA = np.random.rand(100,20)
v = np.random.normal(2,0.01,size=20)
c = []

for i in range(100):
    row_of_A = matrixA[i, :]
    product = row_of_A * v
    sum_of_products = np.sum(product)
    c.append(sum_of_products)

c = np.array(c)

mean_c = np.mean(c)
std_c = np.std(c)

plt.hist(c, bins=5, edgecolor='black')
plt.title("Histogram of vector c")
plt.show()