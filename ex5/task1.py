import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bank_df = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/bank.csv', sep=";")
red_wine_df = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/winequality-red.csv', sep=";")

bank_object = bank_df.select_dtypes(np.object_)
for column in bank_object.columns:
    bank_df[column] = pd.factorize(bank_df[column])[0]

red_wine_object = red_wine_df.select_dtypes(np.object_)
for column in red_wine_object.columns:
    red_wine_df[column] = pd.factorize(red_wine_df[column])[0]

print(bank_df.isna().sum().sum())
print(red_wine_df.isna().sum().sum())

train_ratio = 0.8

Xdata = bank_df[['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
                 'previous', 'poutcome']]
Ydata = bank_df['y']

n = len(bank_df)
train_indices_shuffled = np.random.permutation(n)
train_size_bank = int(n * train_ratio)
train_indices_bank = train_indices_shuffled[:train_size_bank]
test_indices_bank = train_indices_shuffled[train_size_bank:]

Xtrain_bank, Xtest_bank = Xdata.iloc[train_indices_bank], Xdata.iloc[test_indices_bank]
ytrain_bank, ytest_bank = Ydata.iloc[train_indices_bank], Ydata.iloc[test_indices_bank]

Xtrain_bank = (Xtrain_bank - np.mean(Xtrain_bank, axis=0)) / np.std(Xtrain_bank, axis=0)
Xtest_bank = (Xtest_bank - np.mean(Xtest_bank, axis=0)) / np.std(Xtest_bank, axis=0)
ytrain_bank = (ytrain_bank - np.mean(ytrain_bank, axis=0)) / np.std(ytrain_bank, axis=0)
ytest_bank = (ytest_bank - np.mean(ytest_bank, axis=0)) / np.std(ytest_bank, axis=0)

Xwine = red_wine_df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol']]
Ywine = red_wine_df['quality']

n_redWine = len(red_wine_df)
indices_wine_red = np.random.permutation(n_redWine)
train_size_wine_red = int(n_redWine * train_ratio)
train_indices_wine_red = indices_wine_red[:train_size_wine_red]
test_indices_wine_red = indices_wine_red[train_size_wine_red:]

Xtrain_wine_red, Xtest_wine_red = Xwine.iloc[train_indices_wine_red], Xwine.iloc[test_indices_wine_red]
ytrain_wine_red, ytest_wine_red = Ywine.iloc[train_indices_wine_red], Ywine.iloc[test_indices_wine_red]

Xtrain_wine_red = (Xtrain_wine_red - np.mean(Xtrain_wine_red, axis=0)) / np.std(Xtrain_wine_red, axis=0)
Xtest_wine_red = (Xtest_wine_red - np.mean(Xtest_wine_red, axis=0)) / np.std(Xtest_wine_red, axis=0)
ytrain_wine_red = (ytrain_wine_red - np.mean(ytrain_wine_red, axis=0)) / np.std(ytrain_wine_red, axis=0)
ytest_wine_red = (ytest_wine_red - np.mean(ytest_wine_red, axis=0)) / np.std(ytest_wine_red, axis=0)


#######################TASK1!!!#######################
#######################TASK1!!!#######################

def predict(X, beta):
    return np.dot(X, beta)


def lin_gradient(X_batch, y_batch, beta, lambda_):
    y_hat = predict(X_batch, beta)
    error = y_hat - y_batch
    gradient = (2 / len(y_batch)) * (X_batch.T @ error) + 2 * lambda_ * beta
    return gradient


def linear_ridge_regression_mini_batch(Xtrain, ytrain, Xtest, ytest, alpha, lambda_, batch_size, epochs):
    rows, columns = Xtrain.shape
    beta = np.random.randn(columns) * 0.01  # multiplied to prevent some overflows!
    rmse_train = []
    rmse_test = []

    for epoch in range(epochs):
        permutation = np.random.permutation(rows)
        X_shuffled = Xtrain[permutation]
        y_shuffled = ytrain[permutation]

        for i in range(0, rows, batch_size):
            start_index = i
            end_index = i + batch_size
            X_batch = X_shuffled[start_index:end_index]
            y_batch = y_shuffled[start_index:end_index]

            gradient = lin_gradient(X_batch, y_batch, beta, lambda_)

            beta = beta - alpha * gradient

        y_train_pred = predict(Xtrain, beta)
        train_rmse = np.sqrt(np.mean((ytrain - y_train_pred) ** 2))
        rmse_train.append(train_rmse)

        y_test_pred = predict(Xtest, beta)
        test_rmse = np.sqrt(np.mean((ytest - y_test_pred) ** 2))
        rmse_test.append(test_rmse)

    return beta, rmse_train, rmse_test


###################################Logistic Ridge Regression##########

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_probability(X, beta):
    return sigmoid(np.dot(X, beta))


def log_loss(D, theta):
    X, y = D
    probabilities = predict_probability(X, theta)
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))


def logistic_gradient(X_batch, y_batch, beta, lambda_):
    probabilities = predict_probability(X_batch, beta)
    error = probabilities - y_batch
    gradient = (1 / len(y_batch)) * (X_batch.T @ error) + 2 * lambda_ * beta
    return gradient


def logistic_ridge_regression_mini_batch(Xtrain, ytrain, Xtest, ytest, alpha, lambda_, batch_size, epochs):
    rows, columns = Xtrain.shape
    beta = np.random.randn(columns) * 0.01
    log_loss_train = []
    log_loss_test = []

    for epoch in range(epochs):
        permutation = np.random.permutation(rows)
        X_shuffled = Xtrain[permutation]
        y_shuffled = ytrain[permutation]

        for i in range(0, rows, batch_size):
            start_index = i
            end_index = i + batch_size
            X_batch = X_shuffled[start_index:end_index]
            y_batch = y_shuffled[start_index:end_index]

            gradient = logistic_gradient(X_batch, y_batch, beta, lambda_)
            beta = beta - alpha * gradient

        train_log_loss = log_loss((Xtrain, ytrain), beta)
        log_loss_train.append(train_log_loss)

        test_log_loss = log_loss((Xtest, ytest), beta)
        log_loss_test.append(test_log_loss)

    return beta, log_loss_train, log_loss_test


alpha_values = [0.0001, 0.001, 0.01]
lambda_values = [0.1, 1, 10]
batch_size = 50
epochs = 200

for alpha in alpha_values:
    for lambda_ in lambda_values:
        print(f"Training with alpha={alpha}, lambda={lambda_}")
        beta, log_loss_train, log_loss_test = logistic_ridge_regression_mini_batch(
            Xtrain_bank.values, ytrain_bank.values, Xtest_bank.values, ytest_bank.values,
            alpha, lambda_, batch_size, epochs
        )

        plt.figure(figsize=(10, 5))
        plt.plot(log_loss_train, label="Log_loss Train", color="red")
        plt.plot([-l for l in log_loss_test], label="Log_loss Test", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Log_Loss")
        plt.title(f"Training and Test Log_Loss for alpha={alpha}, lambda={lambda_}")
        plt.legend()
        plt.show()

for alpha in alpha_values:
    for lambda_ in lambda_values:
        beta, rmse_train, rmse_test = linear_ridge_regression_mini_batch(
            Xtrain_wine_red.values, ytrain_wine_red.values, Xtest_wine_red.values, ytest_wine_red.values,
            alpha, lambda_, batch_size, epochs
        )

        plt.figure(figsize=(10, 5))
        plt.plot(rmse_train, label="RMSE Train", color="red")
        plt.plot([-r for r in rmse_test], label="RMSE Test", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title(f"Training and Test RMSE for alpha={alpha}, lambda={lambda_}")
        plt.legend()
        plt.show()


###########Task 2 #################
###########Task 2 #################
###########Task 2 #################


def spit_train_into_folds(X, y, k):
    n = len(X)
    indices = np.random.permutation(n)
    fold_size = n // k
    folds_list = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size
        validation_indices = indices[start:end]
        training_indices = np.concatenate([indices[:start], indices[end:]])

        X_train_fold, X_val_fold = X[training_indices], X[validation_indices]
        y_train_fold, y_val_fold = y[training_indices], y[validation_indices]

        folds_list.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))

    return folds_list


def grid_search_kfold_for_rmse(X_train, y_train, alpha_values, lambda_values, batch_size, epochs, k):
    mean_rmse_list = []
    folds = spit_train_into_folds(X_train, y_train, k)

    for alpha in alpha_values:
        for lambda_ in lambda_values:
            rmse_folds = []

            for X_train_fold, y_train_fold, X_val_fold, y_val_fold in folds:
                beta, _, _ = linear_ridge_regression_mini_batch(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    alpha, lambda_, batch_size, epochs
                )
                y_val_pred = predict(X_val_fold, beta)
                rmse = np.sqrt(np.mean((y_val_fold - y_val_pred) ** 2))
                rmse_folds.append(rmse)

            mean_rmse = np.mean(rmse_folds)
            mean_rmse_list.append((alpha, lambda_, mean_rmse))

    return mean_rmse_list


results = grid_search_kfold_for_rmse(Xtrain_wine_red.values, ytrain_wine_red.values, alpha_values, lambda_values,
                                     batch_size,
                                     epochs, 5)


def plot_alpha_lambda_rmse(results, alpha_values, lambda_values):
    data = np.zeros((len(alpha_values), len(lambda_values)))

    for (alpha, lambda_, mean_rmse) in results:
        alpha_idx = alpha_values.index(alpha)
        lambda_idx = lambda_values.index(lambda_)
        data[alpha_idx, lambda_idx] = mean_rmse

    heatmap = plt.pcolor(data, cmap="YlGnBu", shading='auto')

    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x + 0.5, y + 0.5, f'{data[y, x]:.2f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='black', fontsize=10)

    plt.colorbar(heatmap)
    plt.xticks(ticks=np.arange(data.shape[1]) + 0.5, labels=lambda_values)
    plt.yticks(ticks=np.arange(data.shape[0]) + 0.5, labels=alpha_values)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.title("Heatmap with Annotated Values")
    plt.show()


plot_alpha_lambda_rmse(results, alpha_values, lambda_values)

optimal_alpha = 0.001
optimal_lambda = 0.1

beta, rmse_train, rmse_test = linear_ridge_regression_mini_batch(
    Xtrain_wine_red.values, ytrain_wine_red.values, Xtest_wine_red.values, ytest_wine_red.values,
    optimal_alpha, optimal_lambda, batch_size, epochs
)

plt.figure(figsize=(10, 5))
plt.plot(rmse_train, label="RMSE Train", color="blue")
plt.plot([-r for r in rmse_test], label="RMSE Test (negative)", color="green")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"Training and Test RMSE for alpha={optimal_alpha}, lambda={optimal_lambda}")
plt.legend()
plt.show()


##########################
##########################

def grid_search_kfold_for_log_loss(X_train, y_train, alpha_values, lambda_values, batch_size, epochs, k):
    mean_log_loss_list = []
    folds = spit_train_into_folds(X_train, y_train, k)

    for alpha in alpha_values:
        for lambda_ in lambda_values:
            log_loss_folds = []

            for X_train_fold, y_train_fold, X_val_fold, y_val_fold in folds:
                beta, _, _ = logistic_ridge_regression_mini_batch(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    alpha, lambda_, batch_size, epochs
                )

                log_loss1 = log_loss((X_val_fold, y_val_fold), beta)
                log_loss_folds.append(log_loss1)

            mean_log_loss = np.mean(log_loss_folds)
            mean_log_loss_list.append((alpha, lambda_, mean_log_loss))

    return mean_log_loss_list


results1 = grid_search_kfold_for_log_loss(Xtrain_bank.values, ytrain_bank.values, alpha_values, lambda_values,
                                          batch_size,
                                          epochs, 5)


def plot_alpha_lambda_log_loss(results1, alpha_values, lambda_values):
    data1 = np.zeros((len(alpha_values), len(lambda_values)))
    for (alpha, lambda_, mean_log_loss) in results1:
        alpha_idx = alpha_values.index(alpha)
        lambda_idx = lambda_values.index(lambda_)
        data1[alpha_idx, lambda_idx] = mean_log_loss

    heatmap = plt.pcolor(data1, cmap="YlGnBu", shading='auto')

    for y in range(data1.shape[0]):
        for x in range(data1.shape[1]):
            plt.text(x + 0.5, y + 0.5, f'{data1[y, x]:.2f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     color='black', fontsize=10)

    plt.colorbar(heatmap)
    plt.xticks(ticks=np.arange(data1.shape[1]) + 0.5, labels=lambda_values)
    plt.yticks(ticks=np.arange(data1.shape[0]) + 0.5, labels=alpha_values)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.title("Heatmap with Annotated Values")
    plt.show()


plot_alpha_lambda_log_loss(results1, alpha_values, lambda_values)

optimal_alpha_1 = 0.01
optimal_lambda_1 = 0.1
beta1, log_loss_train, log_loss_test = linear_ridge_regression_mini_batch(
    Xtrain_bank.values, ytrain_bank.values, Xtest_bank.values, ytest_bank.values,
    optimal_alpha_1, optimal_lambda_1, batch_size, epochs
)

plt.figure(figsize=(10, 5))
plt.plot(log_loss_train, label="RMSE Train", color="blue")
plt.plot([-l for l in log_loss_test], label="Log_Loss Test (negative)", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Log_Loss")
plt.title(f"Training and Test Log_Loss for alpha={optimal_alpha_1}, lambda={optimal_lambda_1}")
plt.legend()
plt.show()
