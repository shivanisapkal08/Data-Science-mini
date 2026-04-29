#Unemployment in covid

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_1 = pd.read_csv("Unemployment in India.csv")
data_2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

data_1.head()
data_2.head()

data_1.info()
data_2.info()

data_1.columns = data_1.columns.str.strip()
data_2.columns = data_2.columns.str.strip()

data_1['Date'] = pd.to_datetime(data_1['Date'])
data_2['Date'] = pd.to_datetime(data_2['Date'])

data_1.fillna(data_1.mean(numeric_only=True), inplace=True)
data_2.fillna(data_2.mean(numeric_only=True), inplace=True)

data = pd.concat([data_1, data_2], axis=0)
data.reset_index(drop=True, inplace=True)

#Unemployment Rate Over Time (Covid Impact)
plt.figure(figsize=(8,6))
plt.plot(data['Date'], data['Estimated Unemployment Rate (%)'], color='red')
plt.title("Unemployment Rate in India During Covid-19")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.savefig("Unemployment Rate Over Time (Covid Impact).png", dpi=300, bbox_inches='tight')
plt.show()

#State-wise Average Unemployment Rate
state_avg = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)

state_avg.plot(kind='bar')
plt.title("State-wise Average Unemployment Rate")
plt.ylabel("Rate (%)")
plt.savefig("State-wise Average Unemployment Rate.png", dpi=300, bbox_inches='tight')
plt.show()

x = data[['Estimated Labour Participation Rate (%)']]
y = data['Estimated Unemployment Rate (%)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True,random_state=42)

#Linear regression
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

#Final data
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Unemployment Rate")
plt.ylabel("Predicted Unemployment Rate")
plt.title("Actual vs Predicted Unemployment Rate")
plt.show()

import joblib
joblib.dump(model, 'Unemployment_rate.pkl')