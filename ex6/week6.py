import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(7)
X = np.random.normal(loc=1, scale=0.05, size=(100, 1))
psi = np.random.randn(100, 1)
y = 1.3 * X ** 2 + 4.8 * X + 8 + psi

red_wine_df = pd.read_csv('C:/Users/Stefan Koritarev/Desktop/winequality-red.csv', sep=";")
red_wine_object = red_wine_df.select_dtypes(np.object_)
for column in red_wine_object.columns:
    red_wine_df[column] = pd.factorize(red_wine_df[column])[0]

Xwine = red_wine_df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol']]
y_wine = red_wine_df['quality']

scaler = StandardScaler()
Xwine_normalized = scaler.fit_transform(Xwine)

X_train, X_test, y_train, y_test = train_test_split(Xwine_normalized, y_wine, test_size=0.2, random_state=42)


def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    return train_rmse, test_rmse


ridge_params = [0.1, 1, 10]
lasso_params = [0.01, 0.1, 1]
sgd_params = [{'alpha': 0.1, 'max_iter': 1000}, {'alpha': 0.01, 'max_iter': 1000}, {'alpha': 1, 'max_iter': 1000}]

results = {'Model': [], 'Hyperparameter': [], 'Train_RMSE': [], 'Test_RMSE': []}

ols = LinearRegression()
train_rmse, test_rmse = train_model(ols, X_train, y_train, X_test, y_test)
results['Model'].append("OLS")
results['Hyperparameter'].append("None")
results['Train_RMSE'].append(train_rmse)
results['Test_RMSE'].append(test_rmse)

for alpha in ridge_params:
    ridge = Ridge(alpha=alpha)
    train_rmse, test_rmse = train_model(ridge, X_train, y_train, X_test, y_test)
    results['Model'].append("Ridge")
    results['Hyperparameter'].append(f"alpha={alpha}")
    results['Train_RMSE'].append(train_rmse)
    results['Test_RMSE'].append(test_rmse)

for alpha in lasso_params:
    lasso = Lasso(alpha=alpha)
    train_rmse, test_rmse = train_model(lasso, X_train, y_train, X_test, y_test)
    results['Model'].append("LASSO")
    results['Hyperparameter'].append(f"alpha={alpha}")
    results['Train_RMSE'].append(train_rmse)
    results['Test_RMSE'].append(test_rmse)

for param in sgd_params:
    sgd = SGDRegressor(alpha=param['alpha'], max_iter=param['max_iter'], penalty='l2', random_state=42)
    train_rmse, test_rmse = train_model(sgd, X_train, y_train, X_test, y_test)
    results['Model'].append("SGD")
    results['Hyperparameter'].append(f"alpha={param['alpha']}")
    results['Train_RMSE'].append(train_rmse)
    results['Test_RMSE'].append(test_rmse)

results_df = pd.DataFrame(results)
models = results_df['Model'].unique()

for model in models:
    model_data = results_df[results_df['Model'] == model]

    plt.figure(figsize=(8, 6))
    plt.plot(model_data['Hyperparameter'], model_data['Train_RMSE'], marker='o', label="Train RMSE")
    plt.plot(model_data['Hyperparameter'], model_data['Test_RMSE'], marker='x', label="Test RMSE")
    plt.xlabel("Hyperparameter")
    plt.ylabel("RMSE")
    plt.title(f"Train vs Test RMSE for {model}")
    plt.legend()
    plt.grid()
    plt.show()

##########################
##########################

# Ridge Regression Hyperparameter Tuning
ridge_grid = {'alpha': [0.01, 1, 10]}
ridge_gs = GridSearchCV(Ridge(), ridge_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
ridge_gs.fit(X_train, y_train)

# LASSO Regression Hyperparameter Tuning
lasso_grid = {'alpha': [0.01, 0.1, 1]}
lasso_gs = GridSearchCV(Lasso(), lasso_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
lasso_gs.fit(X_train, y_train)

# SGD Regressor Hyperparameter Tuning
sgd_grid = {'alpha': [0.01, 0.1, 1], 'max_iter': [1000]}
sgd_gs = GridSearchCV(SGDRegressor(penalty='l2', random_state=42), sgd_grid, scoring='neg_mean_squared_error', cv=5,
                      verbose=1)
sgd_gs.fit(X_train, y_train)

best_params = {"Ridge": ridge_gs.best_params_,
               "LASSO": lasso_gs.best_params_,
               "SGD": sgd_gs.best_params_
               }

best_ridge_params = ridge_gs.best_params_
best_lasso_params = lasso_gs.best_params_
best_sgd_params = sgd_gs.best_params_

ridge_best = Ridge(alpha=best_ridge_params['alpha'])
lasso_best = Lasso(alpha=best_lasso_params['alpha'])
sgd_best = SGDRegressor(alpha=best_sgd_params['alpha'], max_iter=best_sgd_params['max_iter'], random_state=42)

cv_results = {}

ridge_cv_scores = cross_val_score(ridge_best, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
cv_results["Ridge"] = -ridge_cv_scores

lasso_cv_scores = cross_val_score(lasso_best, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
cv_results["LASSO"] = -lasso_cv_scores

sgd_cv_scores = cross_val_score(sgd_best, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
cv_results["SGD"] = -sgd_cv_scores

plt.figure(figsize=(10, 6))

# Converting dictionary keys to a list for compatibility
labels = list(cv_results.keys())
data = list(cv_results.values())

plt.boxplot(data, labels=labels)
plt.ylabel("Mean Squared Error")
plt.title("Cross-Validation MSE for Different Models")
plt.grid()
plt.show()

###################Task 2 ##########

degrees = [1, 2, 7, 10, 16, 100]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

plt.figure(figsize=(12, 8))

for i, degree in enumerate(degrees):
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_poly, y)

    X_vals = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_vals_poly = poly_features.transform(X_vals)
    y_vals = reg.predict(X_vals_poly)

    plt.plot(X_vals, y_vals, color=colors[i], label=f'Degree {degree}')

# Plotting the actual data points (scatter plot)
plt.scatter(X, y, color='black', label='Data')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Prediction Curves for Polynomial Regression")
plt.legend()
plt.show()

### task b

degree = 10
lambdas_list = [0, 10 - 6, 10 - 2, 1]

poly_features = PolynomialFeatures(degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

plt.figure(figsize=(12, 8))

for lamda in lambdas_list:
    ridge_reg = Ridge(alpha=lamda)
    ridge_reg.fit(X_poly, y)

    X_vals = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_vals_poly = poly_features.transform(X_vals)
    y_vals = ridge_reg.predict(X_vals_poly)

    label = f"λ = {lamda:.0e}"
    plt.plot(X_vals, y_vals, label=label)

# Plotting the actual data points (scatter plot)
plt.scatter(X, y, color='black', label='Data', alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Effect of Regularization in Ridge Regression")
plt.legend()
plt.grid()
plt.show()
