import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_columns', 500)

df_airfare = pd.read_csv("https://users.stat.ufl.edu/~winner/data/airq402.dat", sep=r'\s+')

df_airfare.columns = ["City1", "City2", "Average_Fare", "Distance", "Average_weekly_passengers",
                      "Market_leading_airline", "Market_share_1", "Average_fare_2",
                      "Low_price_airline", "Market_share_2", "Price"]

print(df_airfare.isnull().sum().sum())
df_numeric_airfare = df_airfare.select_dtypes(np.number)
df_object_airfare = df_airfare.select_dtypes(np.object_)
df_airfare_newNum = pd.get_dummies(df_object_airfare, dtype='int')
df_airfare_allNum = pd.concat([df_numeric_airfare, df_airfare_newNum], axis=1)
print(df_airfare_allNum.isnull().sum().sum())

train_ratio = 0.8
n = len(df_airfare_allNum)
indices_airfare_shuffled = np.random.permutation(n)
train_size_airfare = int(n * train_ratio)
train_indices_airfare = indices_airfare_shuffled[:train_size_airfare]
test_indices_airfare = indices_airfare_shuffled[train_size_airfare:]

Xtrain_airfare, Xtest_airfare = df_airfare_allNum.iloc[train_indices_airfare], df_airfare_allNum.iloc[
    test_indices_airfare]

df_wine_red = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/winequality-red.csv', sep=";")
print(df_wine_red.isnull().sum())

n_redWine = len(df_wine_red)
indices_wine_red = np.random.permutation(n_redWine)
train_size_wine_red = int(n_redWine * train_ratio)
train_indices_wine_red = indices_wine_red[:train_size_wine_red]
test_indices_wine_red = indices_wine_red[train_size_wine_red:]

Xtrain_wine_red, Xtest_wine_red = df_wine_red.iloc[train_indices_wine_red], df_wine_red.iloc[test_indices_wine_red]

##########################################################
#########################################


df_wine_white = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/winequality-white.csv', sep=";")
print(df_wine_white.isnull().sum().sum())
print(df_wine_white.dtypes)

df_wine_white['alcohol'] = pd.to_numeric(df_wine_white['alcohol'], errors='coerce')
df_wine_white['alcohol'] = df_wine_white['alcohol'].fillna(df_wine_white['alcohol'].median())

Xdata = df_wine_white[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']]
Ydata = df_wine_white['quality']

n_whiteWine = len(df_wine_white)
indices_wine_white = np.random.permutation(n_whiteWine)
train_size_wine_white = int(n_whiteWine * train_ratio)
train_indices_wine_white = indices_wine_white[:train_size_wine_white]
test_indices_wine_white = indices_wine_white[train_size_wine_white:]

Xtrain_wine_white, Xtest_wine_white = Xdata.iloc[train_indices_wine_white], Xdata.iloc[test_indices_wine_white]
ytrain_wine_white, ytest_wine_white = Ydata.iloc[train_indices_wine_white], Ydata.iloc[test_indices_wine_white]

Xtrain_wine_white = (Xtrain_wine_white - Xtrain_wine_white.mean()) / Xtrain_wine_white.std()
Xtest_wine_white = (Xtest_wine_white - Xtest_wine_white.mean()) / Xtest_wine_white.std()

Dtrain = (Xtrain_wine_white, ytrain_wine_white)
Dtest = (Xtest_wine_white, ytest_wine_white)


def predict(X, beta):
    return np.dot(X, beta)


# Loss function
def least_square_loss(X, y, beta):
    X, y, beta = np.array(X), np.array(y), np.array(beta)
    y_pred = predict(X, beta)
    squared_errors = (y - y_pred) ** 2
    loss = np.sum(squared_errors)
    return loss

imax = 1000

def gradient(X, y, beta):
    X, y, beta = np.array(X), np.array(y), np.array(beta)
    yhat = predict(X, beta)
    return -2 * X.T.dot(y - yhat)


def minimizeGD(f, x0, alpha, imax, epsilon, Xtest, ytest):
    beta = np.array(x0, dtype=float)
    X, y = f
    loss_diffs = []
    test_rmse = []

    for i in range(imax):
        current_loss = least_square_loss(X, y, beta)
        d = gradient(X, y, beta)
        beta_new = beta - alpha * d
        new_loss = least_square_loss(X, y, beta_new)

        abs_diff = abs(current_loss - new_loss)
        loss_diffs.append(abs_diff)

        y_pred_test = predict(Xtest, beta_new)
        rmse_values = np.sqrt(np.mean((ytest - y_pred_test) ** 2))
        test_rmse.append(rmse_values)

        if abs_diff < epsilon:
            break

        beta = beta_new

    print("Not converged within imax iterations")
    return beta, loss_diffs, test_rmse


def learn_lin_reg_GD(Dtrain, Dtest, alphas, imax, epsilon):
    global beta_opt
    Xtrain, ytrain = Dtrain
    Xtest, ytest = Dtest

    for alpha in alphas:
        beta_0 = np.zeros(Xtrain.shape[1])

        beta_opt, loss_diffs, test_rmse = minimizeGD((Xtrain.values, ytrain.values), beta_0, alpha, imax, epsilon,
                                                     Xtest.values, ytest.values)

        plt.figure(figsize=(12, 5))
        plt.plot(loss_diffs, label=f'α={alpha}')
        plt.xlabel('Iteration')
        plt.ylabel('|f(xi−1) − f(xi)|')
        plt.title(f'Convergence of Loss Difference for α={alpha}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.plot(test_rmse, label=f'α={alpha}')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title(f'RMSE Over Iterations for α={alpha}')
        plt.legend()
        plt.show()

    return beta_opt


alphas = [0.01, 0.001, 0.0001]
epsilon = 1e-6

beta_predicted = learn_lin_reg_GD(
    (Xtrain_wine_white, ytrain_wine_white),
    (Xtest_wine_white, ytest_wine_white),
    alphas,
    imax,
    epsilon
)
print("Optimized beta:", beta_predicted)


########## B): ##########
########## B): ##########


def minimizeGD_dynamic(f, x0, imax, epsilon, step_length, Xtest, ytest, **kwargs):
    beta = np.array(x0, dtype=float)
    X, y = f
    loss_diffs = []
    test_rmse = []

    for i in range(imax):
        grad = gradient(X, y, beta)
        d = -grad

        alpha = step_length(lambda b: least_square_loss(X, y, b), beta, d, **kwargs)
        beta_new = beta + alpha * d
        loss_diff = np.abs(least_square_loss(X, y, beta_new) - least_square_loss(X, y, beta))
        loss_diffs.append(loss_diff)

        rmse_value = rmse(Xtest, ytest, beta_new)
        test_rmse.append(rmse_value)

        if loss_diff < epsilon:
            break

        beta = beta_new

    print("Not converged within imax iterations")
    return beta, loss_diffs, test_rmse


def learn_lin_reg_GD_dynamic(Dtrain, imax, epsilon, step_length, Xtest, ytest, **kwargs):
    Xtrain, ytrain = Dtrain
    beta_0 = np.zeros(Xtrain.shape[1])
    beta_opt, loss_diffs, test_rmse = minimizeGD_dynamic(
        (Xtrain.values, ytrain.values), beta_0, imax, epsilon, step_length,
        Xtest=Xtest.values, ytest=ytest.values, **kwargs
    )
    return beta_opt, loss_diffs, test_rmse


def rmse(X, y, beta):
    y_pred = predict(X, beta)
    return np.sqrt(np.mean((y - y_pred) ** 2))


def steplength_armijo(f, x, d, delta=0.5):
    alpha = 1
    while f(x) - f(x + alpha * d) < alpha * delta * np.dot(d, d):
        alpha /= 2
    return alpha


def steplength_bolddriver(f, x, d, alpha_old, alpha_plus=1.1, alpha_minus=0.5):
    alpha = alpha_old * alpha_plus
    while f(x) - f(x + alpha * d) <= 0:
        alpha *= alpha_minus
    return alpha


# Example usage with Armijo step length
beta_predicted_armijo, losses_armijo, test_rmse_armijo = learn_lin_reg_GD_dynamic(
    (Xtrain_wine_white, ytrain_wine_white),
    imax=imax,
    epsilon=epsilon,
    step_length=steplength_armijo,
    delta=0.5,
    Xtest=Xtest_wine_white,
    ytest=ytest_wine_white
)
print("Optimized beta (Armijo):", beta_predicted_armijo)
plt.plot(losses_armijo, label="Loss Difference (Armijo)")

# Bold Driver example
beta_predicted_bolddriver, losses_bolddriver, test_rmse_bolddriver = learn_lin_reg_GD_dynamic(
    (Xtrain_wine_white, ytrain_wine_white),
    imax=imax,
    epsilon=epsilon,
    step_length=steplength_bolddriver,
    alpha_old=1,
    alpha_plus=1.1,
    alpha_minus=0.5,
    Xtest=Xtest_wine_white,
    ytest=ytest_wine_white
)
print("Optimized beta (Bold Driver):", beta_predicted_bolddriver)
plt.plot(losses_bolddriver, label="Loss Difference (Bold Driver)")
plt.xlabel("Iterations")
plt.ylabel("Loss Difference |f(x_{i-1}) - f(x_i)|")
plt.legend()
plt.title("Convergence of Loss Difference with Armijo and Bold Driver")
plt.show()

plt.plot(test_rmse_armijo, label="Test RMSE (Armijo)")
plt.plot(test_rmse_bolddriver, label="Test RMSE (Bold Driver)")
plt.xlabel("Iteration")
plt.ylabel("Test RMSE")
plt.legend()
plt.title("Test RMSE with Armijo and Bold Driver")
plt.show()
