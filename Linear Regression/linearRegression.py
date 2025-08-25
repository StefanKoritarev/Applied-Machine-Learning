import numpy as np
import matplotlib.pyplot as plt

matrixA = np.random.normal(2, 0.01, size=(100, 2))
matrixB = np.random.normal(2, 0.1, size=(100, 2))
matrixC = np.random.normal(2, 1, size=(100, 2))


def learn_simple_linreg(matrix):
    x_values = matrix[:, 0]
    y_values = matrix[:, 1]

    x_avg = np.mean(x_values)
    y_avg = np.mean(y_values)

    upper_sum = np.sum((x_values - x_avg) * (y_values - y_avg))
    lower_sum = np.sum((x_values - x_avg) ** 2)
    b1_estimated = upper_sum / lower_sum

    b0_estimated = y_avg - b1_estimated * x_avg
    return b0_estimated, b1_estimated

b0_A, b1_A = learn_simple_linreg(matrixA)
b0_B, b1_B = learn_simple_linreg(matrixB)
b0_C, b1_C = learn_simple_linreg(matrixC)

def predict_simple_linreg(x, b0_estimated, b1_estimated):
    y_estimated = b0_estimated + x * b1_estimated
    return y_estimated

x_values_A = matrixA[:, 0]
predicted_y_A = predict_simple_linreg(x_values_A, b0_A, b1_A)

x_values_B = matrixB[:, 0]
predicted_y_B = predict_simple_linreg(x_values_B, b0_B, b1_B)

x_values_C = matrixC[:, 0]
predicted_y_C = predict_simple_linreg(x_values_C, b0_C, b1_C)

plt.figure(figsize=(10, 8))

plt.scatter(x_values_A, matrixA[:, 1], color='blue', label='Matrix A (σ = 0.01)', alpha=1.0)
plt.plot(np.sort(x_values_A), np.sort(predicted_y_A), color='blue', linestyle='--', linewidth=2)

plt.scatter(x_values_B, matrixB[:, 1], color='green', label='Matrix B (σ = 0.1)', alpha=0.6)
plt.plot(np.sort(x_values_B), np.sort(predicted_y_B), color='green', linestyle='--', linewidth=2)

plt.scatter(x_values_C, matrixC[:, 1], color='red', label='Matrix C (σ = 1)', alpha=0.4)
plt.plot(np.sort(x_values_C), np.sort(predicted_y_C), color='red', linestyle='--', linewidth=2)

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot of Matrices A, B, and C with Predicted Lines')
plt.legend()
plt.show()
