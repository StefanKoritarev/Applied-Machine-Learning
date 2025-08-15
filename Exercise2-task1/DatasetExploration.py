import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', 100)

df = pd.read_csv('C:/Users/Stefan Koritarev/Downloads/GasPrices.csv')
print(df.describe())

df_grouped = df.groupby('Name')[['Price', 'Income', 'Pumps']].agg('mean')

selected_names = ['Shell', '7-Eleven', 'Gulf']
filtered_df = df[df['Name'].isin(selected_names)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
filtered_df.boxplot(column='Price', by='Name', ax=axes[0])
axes[0].set_title('Boxplot of Price by Selected Names')
axes[0].set_ylabel('Price')
axes[0].set_xlabel('Name')

filtered_df.boxplot(column='Pumps', by='Name', ax=axes[1])
axes[1].set_title('Boxplot of Pumps by Selected Names')
axes[1].set_ylabel('Pumps')
axes[1].set_xlabel('Name')

filtered_df.boxplot(column='Gasolines', by='Name', ax=axes[2])
axes[2].set_title('Boxplot of Gasolines by Selected Names')
axes[2].set_ylabel('Gasolines')
axes[2].set_xlabel('Name')

plt.suptitle('Boxplots for Selected Gas Stations', fontsize=16)
plt.show()

price = df['Price'].values
income = df['Income'].values

def learn_simple_linreg(x_values, y_values):
    x_avg = np.mean(x_values)
    y_avg = np.mean(y_values)

    upper_sum = np.sum((x_values - x_avg) * (y_values - y_avg))
    lower_sum = np.sum((x_values - x_avg) ** 2)
    b1_estimated = upper_sum / lower_sum

    b0_estimated = y_avg - b1_estimated * x_avg
    return b0_estimated, b1_estimated

b0, b1 = learn_simple_linreg(price, income)

def predict_simple_linreg(x, b0_estimated, b1_estimated):
    y_estimated = b0_estimated + x * b1_estimated
    return y_estimated

predicted_income = predict_simple_linreg(price, b0, b1)

plt.figure(figsize=(10, 8))
plt.scatter(price, income, color='blue', label='Actual Values')
plt.plot(np.sort(price), np.sort(predicted_income), color='red', linestyle='--', linewidth=2, label='Predicted Line')
plt.xlabel('Price')
plt.ylabel('Income')
plt.title('Price vs Income with Predicted Linear Regression Line')
plt.legend()
plt.show()


min_income = np.min(income)
max_income = np.max(income)

income_normalized = (income - min_income) / (max_income - min_income)
b0_normalized, b1_normalized = learn_simple_linreg(price, income_normalized)
predicted_income_normalized = predict_simple_linreg(price, b0_normalized, b1_normalized)

plt.subplot(1, 2, 1)
plt.scatter(price, income, color='blue', label='Actual (Price vs Income)')
plt.plot(np.sort(price), np.sort(predicted_income), color='red', linestyle='--', linewidth=2, label='Predicted Line')
plt.xlabel('Price')
plt.ylabel('Income')
plt.title('Price vs Income (Original)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(price, income_normalized, color='green', label='Actual (Price vs Normalized Income)')
plt.plot(np.sort(price), np.sort(predicted_income_normalized), color='red', linestyle='--', linewidth=2,
         label='Predicted Line (Normalized)')
plt.xlabel('Price')
plt.ylabel('Normalized Income')
plt.title('Price vs Normalized Income')
plt.legend()
plt.tight_layout()
plt.show()

print("Original Income Range:", income.min(), "-", income.max())
print("Normalized Income Range:", income_normalized.min(), "-", income_normalized.max())
