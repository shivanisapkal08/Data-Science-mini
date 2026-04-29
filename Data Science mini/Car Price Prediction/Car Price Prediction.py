#Car price prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("car data.csv")
df.head()

df.info()

df.describe()

df.isnull().sum()

#Price distribution
plt.figure(figsize=(8,6))
sns.histplot(df['Selling_Price'], kde=True)
plt.title("Selling Price Distribution")
plt.savefig("selling_price_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

#Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.show()

df.drop('Car_Name', axis=1, inplace=True)

#Feature engineering
df['Car_Age'] = 2024 - df['Year']
df.drop('Year', axis=1, inplace=True)

le = LabelEncoder()

df['Fuel_Type'] = le.fit_transform(df['Fuel_Type'])
df['Selling_type'] = le.fit_transform(df['Selling_type'])
df['Transmission'] = le.fit_transform(df['Transmission'])

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.dtypes)

#Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Linear Regression Performance")
print("MAE:", mean_absolute_error(y_test, y_pred_lr))
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

#Random forest regressor
rf = RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("Random Forest Performance")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

feature_importance = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)

feature_importance.plot(kind='bar', figsize=(8,4))
plt.title("Feature Importance")
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# Example values:
# Present_Price, Kms_Driven, Fuel_Type, Selling_type, Owner, Transmission

new_car = np.array([[45000, 0, 1, 1, 0, 5, 3]])  

predicted_price = rf.predict(new_car)
print("Predicted Car Price:", predicted_price[0])
