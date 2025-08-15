import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)

bank_df = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/bank.csv', sep=";")
full_bank_df = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/bank-full.csv', sep=";")

occupancy_df = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/datatest.txt', delimiter=",")
occupancy_df_1 = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/datatest2.txt', delimiter=",")
occupancy_df_2 = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/datatraining.txt', delimiter=",")

print("bank_df:", bank_df.shape)
print("full_bank_df:", full_bank_df.shape)
print("occupancy_df:", occupancy_df.shape)
print("occupancy_df_1:", occupancy_df_1.shape)
print("occupancy_df_2:", occupancy_df_2.shape)

bank_object = bank_df.select_dtypes(np.object_)
for column in bank_object.columns:
    bank_df[column] = pd.factorize(bank_df[column])[0]
print(bank_df.isna().sum().sum())

occupancy_df_2_object = occupancy_df_2.select_dtypes(np.object_)
for column in occupancy_df_2_object.columns:
    occupancy_df_2[column] = pd.factorize(occupancy_df_2_object[column])[0]
print(occupancy_df_2.isna().sum().sum())

###Splitting the data##

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

X_occupancy = occupancy_df_2[['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
y_occupancy = occupancy_df_2['Occupancy']

n1 = len(occupancy_df_2)
train_indices_shuffled = np.random.permutation(n1)
train_size_occupancy = int(n1 * train_ratio)
train_indices_occupancy = train_indices_shuffled[:train_size_occupancy]
test_indices_occupancy = train_indices_shuffled[train_size_occupancy:]
Xtrain_occupancy, Xtest_occupancy = X_occupancy.iloc[train_indices_occupancy], X_occupancy.iloc[test_indices_occupancy]
ytrain_occupancy, ytest_occupancy = y_occupancy.iloc[train_indices_occupancy], y_occupancy.iloc[test_indices_occupancy]

Dtrain_bank = (Xtrain_bank, ytrain_bank)
Dtrain_occupancy = (Xtrain_occupancy, ytrain_occupancy)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_probability(X, beta):
    return sigmoid(np.dot(X, beta))


def log_likelihood(Dtrain, beta):
    X, y = Dtrain
    temp_probability = predict_probability(X, beta)
    epsilon = 1e-15
    temp_probability = np.clip(temp_probability, epsilon, 1 - epsilon)
    return np.sum(y * np.log(temp_probability) + (1 - y) * np.log(1 - temp_probability))


def gradient_for_xi_yi(x_i, y_i, beta):
    probability = predict_probability(x_i, beta)
    return (y_i - probability) * x_i


def gradient_ascend_stochastic(Dtrain, theta, initial_step_size, imax, tolerance=1e-6):
    X, y = Dtrain
    X = np.array(X)
    y = np.array(y)
    alpha = initial_step_size
    log_losses_on_test_set = []
    log_likelihood_list = []
    n_rows = X.shape[0]

    theta_for_log_loss = np.copy(theta)

    for epoch in range(imax):
        i = np.random.randint(n_rows)
        x_i = X[i]
        y_i = y[i]

        grad = gradient_for_xi_yi(x_i, y_i, theta)
        grad_for_log_loss = gradient_for_xi_yi(x_i, y_i, theta_for_log_loss)

        # Here I have two variants -> one is with dynamically adjust alpha and the other is the fixed rate
        # alpha = steplength_bolddriver(lambda beta_tmp: log_likelihood(Dtrain, beta_tmp), theta, grad, alpha)
        alpha = learning_rate_init

        theta = theta + alpha * grad

        log_likelihood_current = log_likelihood(Dtrain, theta)
        log_likelihood_list.append(log_likelihood_current)

        theta_for_log_loss = theta_for_log_loss - alpha * grad_for_log_loss

        # test_log_loss = log_loss((Xtest_occupancy_intercept, ytest_occupancy), theta_for_log_loss)
        test_log_loss = log_loss((Xtest_bank_intercept, ytest_bank), theta_for_log_loss)
        log_losses_on_test_set.append(test_log_loss)

        if np.linalg.norm(grad) < tolerance:
            print(f'Converged at epoch {epoch}')
            return theta, log_likelihood_list, log_losses_on_test_set
    raise Exception("Algorithm did not converge within the maximum number of epochs")


def log_loss(Dtest, theta):
    X, y = Dtest
    probabilities = predict_probability(X, theta)
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
    return np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))


def steplength_bolddriver(f, x, d, alpha_old, alpha_plus=1.1, alpha_minus=0.5, max_iter=10):
    alpha = alpha_old * alpha_plus
    for _ in range(max_iter):
        if f(x) - f(x + alpha * d) > 0:
            break
        alpha *= alpha_minus
    return alpha


############### TASK 2 ############################ TASK 2 ############################ TASK 2 ############################ TASK 2 ############################ TASK 2 #############
############### TASK 2 ############################ TASK 2 ############################ TASK 2 ############################ TASK 2 ############################ TASK 2 #############


def stepsize_adagrad(grad, h, initial_stepsize):
    h += grad ** 2
    learning_rate = initial_stepsize / (np.sqrt(h) + 1e-8)
    return learning_rate, h


def gradient_ascent_stochastic_adagrad(Dtrain, theta, initial_stepsize, imax, tolerance=1e-6):
    X, y = Dtrain
    X = np.array(X)
    y = np.array(y)

    h = np.zeros_like(theta)

    theta_for_minimizing_loss = np.copy(theta)

    delta_log_likelihoods = []
    test_log_losses = []

    n_rows = X.shape[0]
    for epoch in range(imax):
        i = np.random.randint(n_rows)
        x_i = X[i]
        y_i = y[i]

        grad = gradient_for_xi_yi(x_i, y_i, theta)
        grad_for_minimizing_loss = gradient_for_xi_yi(x_i, y_i, theta_for_minimizing_loss)

        step_sizes, h = stepsize_adagrad(grad, h, initial_stepsize)
        theta = theta + step_sizes * grad

        log_likelihood_current = log_likelihood(Dtrain, theta)
        delta_log_likelihoods.append(log_likelihood_current)

        theta_for_minimizing_loss = theta_for_minimizing_loss - step_sizes * grad_for_minimizing_loss

        test_log_loss = log_loss((Xtest_occupancy_intercept, ytest_occupancy), theta_for_minimizing_loss)
        # test_log_loss = log_loss((Xtest_bank_intercept, ytest_bank), theta_for_minimizing_loss)
        test_log_losses.append(test_log_loss)

        if np.linalg.norm(grad) < tolerance:
            print(f'Converged at epoch {epoch}')
            return theta, delta_log_likelihoods, test_log_losses

    raise Exception("Algorithm did not converge within the maximum number of epochs")


Xtrain_bank_intercept = np.hstack((np.ones((Xtrain_bank.shape[0], 1)), Xtrain_bank))
Xtest_bank_intercept = np.hstack((np.ones((Xtest_bank.shape[0], 1)), Xtest_bank))
Dtrain_bank = (Xtrain_bank_intercept, ytrain_bank)
Dtest_bank = (Xtest_bank_intercept, ytest_bank)
initial_beta_bank = np.zeros(Xtrain_bank_intercept.shape[1])

Xtrain_occupancy_intercept = np.hstack((np.ones((Xtrain_occupancy.shape[0], 1)), Xtrain_occupancy))
Xtest_occupancy_intercept = np.hstack((np.ones((Xtest_occupancy.shape[0], 1)), Xtest_occupancy))
Dtrain_occupancy = (Xtrain_occupancy_intercept, ytrain_occupancy)
Dtest_occupancy = (Xtest_occupancy_intercept, ytest_occupancy)
initial_beta_occupancy = np.zeros(Xtrain_occupancy_intercept.shape[1])

learning_rate_init = 0.01
imax = 1000
tolerance = 1e-6

# Train the model(task 1)
beta_hat, log_likelihood_diffs_epoch, test_log_losses = gradient_ascend_stochastic(Dtrain_bank, initial_beta_bank,
                                                                                   learning_rate_init, imax, tolerance)

# beta_hat, log_likelihood_diffs_epoch, test_log_losses = gradient_ascend_stochastic(Dtrain_occupancy, initial_beta_occupancy,
# learning_rate_init, imax, tolerance)


# Train the AdaGrad model(task 2)
# beta_hat_adagrad, delta_log_likelihoods_adagrad, test_log_losses_adagrad = gradient_ascent_stochastic_adagrad(
#     Dtrain_bank, initial_beta_bank, learning_rate_init, imax, tolerance
# )

beta_hat_adagrad, delta_log_likelihoods_adagrad, test_log_losses_adagrad = gradient_ascent_stochastic_adagrad(
    Dtrain_occupancy, initial_beta_occupancy, learning_rate_init, imax, tolerance)

log_likelihood_diffs = [abs(delta_log_likelihoods_adagrad[i] - delta_log_likelihoods_adagrad[i - 1]) for i in
                        range(1, len(delta_log_likelihoods_adagrad))]

log_likelihood_diffs_task1 = [abs(log_likelihood_diffs_epoch[i] - log_likelihood_diffs_epoch[i - 1]) for i in
                              range(1, len(log_likelihood_diffs_epoch))]

#####TASK 1 plot
#####TASK 1 plot

plt.figure(figsize=(8, 6))
plt.plot(log_likelihood_diffs_task1, label='Log_likelihood_diffs')
plt.xlabel('Epoch')
plt.ylabel('Sum of Absolute Differences in Log-Likelihood')
plt.title('Convergence of Log-Likelihood')
plt.legend()
plt.show()

# Plot log-loss on the test set
plt.figure(figsize=(8, 6))
plt.plot(test_log_losses, label='Log_loss on test set', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Log-Loss')
plt.legend()
plt.tight_layout()
plt.show()

######TASK 2####
######TASK 2####


plt.figure(figsize=(8, 6))
plt.plot(log_likelihood_diffs, label='Log_likelihood (AdaGrad)')
plt.xlabel('Epoch')
plt.ylabel('Sum of Absolute Differences in Log-Likelihood')
plt.title('Convergence of Log-Likelihood - AdaGrad')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(test_log_losses_adagrad, label='Log-Loss on Test Set (AdaGrad)', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Log-Loss')
plt.title('Log-Loss on Test Set Over Iterations - AdaGrad')
plt.legend()
plt.show()
