import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Xdata = pd.read_csv('C:/Users/Stefan Koritarev/Downloads/GasPrices.csv')
print(Xdata.columns)

df_forPrediction = Xdata[['Price', 'Pumps', 'Gasolines', 'Income']]
Xdata = df_forPrediction[['Price', 'Pumps', 'Gasolines']]
Ydata = df_forPrediction['Income']
Ydata = (Ydata - Ydata.min()) / (Ydata.max() - Ydata.min())

train_ratio = 0.8
n = len(Xdata)
indices = np.random.permutation(n)

train_size = int(n * train_ratio)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

Xtrain, Xtest = Xdata.iloc[train_indices], Xdata.iloc[test_indices]
Ytrain, Ytest = Ydata.iloc[train_indices], Ydata.iloc[test_indices]
Ytrain = Ytrain.values.flatten()

Xtrain_b = np.c_[np.ones((Xtrain.shape[0], 1)), Xtrain]
Xtest_b = np.c_[np.ones((Xtest.shape[0], 1)), Xtest]

Dtrain = (Xtrain_b, Ytrain)


def rowElim(M, i, j, x):
    M[i] = [a + x * b for a, b in zip(M[i], M[j])]


def to_upperTrForm(M):
    row, column = 0, 0
    rows, cols = len(M), len(M[0])

    while row < rows and column < cols:
        if M[row][column] == 0:
            # Finding a row below the current one with a non-zero value in the same column
            for r in range(row + 1, rows):
                if M[r][column] != 0:
                    rowElim(M, row, r, 1)
                    break

        # Skipping column this column is only zeros
        if M[row][column] == 0:
            column += 1
            continue

        # Pivot is now non-zero; performing elimination below this row
        pivot = M[row][column]
        for r in range(row + 1, rows):
            if M[r][column] != 0:
                rowElim(M, r, row, -M[r][column] / pivot)
        row += 1
        column += 1


def back_substitution(A, b):
    n = len(b)
    beta = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = np.dot(A[i][i + 1:], beta[i + 1:])
        beta[i] = (b[i] - s) / A[i][i]
    return beta


def gauss_elimination_LinearReg(A, b):
    # Combine A and b into an augmented matrix M
    M = np.column_stack((A, b))
    to_upperTrForm(M)
    # Extract the modified matrix A and vector b after elimination
    A_upper = M[:, :-1]  # extracting all columns of M except from the last one
    b_upper = M[:, -1]  # extracting the last column of M (the modified b)

    beta = back_substitution(A_upper, b_upper)
    return beta


def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            sum_ = np.dot(L[i, :j], L[j, :j])
            if j == i:
                L[i, i] = np.sqrt(A[i, i] - sum_)
            else:
                L[i, j] = (A[i, j] - sum_) / L[j, j]
    return L


def forward_substitution(L, b):
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        s = np.dot(L[i][:i], z[:i])
        z[i] = (b[i] - s) / L[i][i]
    return z


def qr_decomp(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    for i, column in enumerate(A.T):
        Q[:, i] = column
        for prev in Q.T[:i]:
            Q[:, i] -= (prev @ column) / (prev @ prev) * prev
    Q /= np.linalg.norm(Q, axis=0)
    R = Q.T @ A
    return Q, R


def qr_decomposition_solve(X, y):
    Q, R = qr_decomp(X)
    QTy = np.dot(Q.T, y)
    beta = back_substitution(R, QTy)
    return beta


def learn_LinReg_NormEq_gauss(Dtrain):
    Xtrain_b, Ytrain = Dtrain
    A = (Xtrain_b.T).dot(Xtrain_b)
    b = (Xtrain_b.T).dot(Ytrain)
    beta = gauss_elimination_LinearReg(A, b)
    return beta


def learn_LinReg_NormEq_cholesky(Dtrain):
    Xtrain_b, Ytrain = Dtrain
    A = (Xtrain_b.T).dot(Xtrain_b)
    b = (Xtrain_b.T).dot(Ytrain)
    L = cholesky_decomposition(A)
    z = forward_substitution(L, b)
    beta = back_substitution(L.T, z)
    return beta


def learn_LinReg_NormEq_QR(Dtrain):
    Xtrain_b, Ytrain = Dtrain
    beta = qr_decomposition_solve(Xtrain_b, Ytrain)
    return beta


Dtrain_b = (Xtrain_b, Ytrain)
beta_gauss = learn_LinReg_NormEq_gauss(Dtrain_b)
beta_cholesky = learn_LinReg_NormEq_cholesky(Dtrain_b)
beta_qr = learn_LinReg_NormEq_QR(Dtrain_b)


def predict_LinReg(Xtest, beta):
    return Xtest.dot(beta)


Ypred_gauss = predict_LinReg(Xtest_b, beta_gauss)
Ypred_cholesky = predict_LinReg(Xtest_b, beta_cholesky)
Ypred_qr = predict_LinReg(Xtest_b, beta_qr)

residuals_gauss = np.abs(Ytest.values.flatten() - Ypred_gauss)
residuals_cholesky = np.abs(Ytest.values.flatten() - Ypred_cholesky)
residuals_qr = np.abs(Ytest.values.flatten() - Ypred_qr)

average_residual_gauss = np.mean(residuals_gauss)
average_residual_cholesky = np.mean(residuals_cholesky)
average_residual_qr = np.mean(residuals_qr)

N = len(Ytest)
RMSE_gauss = np.sqrt(np.sum((Ytest.values.flatten() - Ypred_gauss) ** 2) / N)
RMSE_cholesky = np.sqrt(np.sum((Ytest.values.flatten() - Ypred_cholesky) ** 2) / N)
RMSE_qr = np.sqrt(np.sum((Ytest.values.flatten() - Ypred_qr) ** 2) / N)

plt.figure(figsize=(13, 5))

plt.subplot(1, 3, 1)
plt.scatter(Ytest, residuals_gauss, color='red', alpha=0.6)
plt.title("Residuals vs True Values (Gaussian Elimination)")
plt.xlabel("True Values (Ytest)")
plt.ylabel("Residuals")

plt.subplot(1, 3, 2)
plt.scatter(Ytest, residuals_cholesky, color='blue', alpha=0.5)
plt.title("Residuals vs True Values (Cholesky)")
plt.xlabel("True Values (Ytest)")
plt.ylabel("Residuals")

plt.subplot(1, 3, 3)
plt.scatter(Ytest, residuals_qr, color='green', alpha=0.5)
plt.title("Residuals vs True Values (QR Decomposition)")
plt.xlabel("True Values (Ytest)")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()

print("Performance Metrics for Each Model:")
print("\nGaussian Elimination:")
print(f"Average Residual: {average_residual_gauss:.2f}")
print(f"Root Mean Square Error (RMSE): {RMSE_gauss:.2f}\n")

print("Cholesky Decomposition:")
print(f"Average Residual: {average_residual_cholesky:.2f}")
print(f"Root Mean Square Error (RMSE): {RMSE_cholesky:.2f}\n")

print("QR Decomposition:")
print(f"Average Residual: {average_residual_qr:.2f}")
print(f"Root Mean Square Error (RMSE): {RMSE_qr:.2f}")
