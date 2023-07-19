import pandas as pd

# from sklearn.ensemble import RandomForestClassifier
# print(df.columns)
# print(df.iloc[1])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.linear_model import r
# Load the transaction dataset
# data = pd.read_csv("transaction_dataset.csv")
data= pd.read_csv('Fraud.csv')
data = data.dropna()
# data = data.drop(['step'],axis=1)
# data = data[[ 'type', 'amount','isFraud', 'isFlaggedFraud']]

# data['type'] = data['type'].apply(lambda x: 0 if x=='CASH_IN' else x)
data['type'] = data['type'].apply(lambda x: 1 if x=='CASH_OUT' else x)
# data['type'] = data['type'].apply(lambda x: 3 if x=='PAYMENT' else x)
data['type'] = data['type'].apply(lambda x: 1 if x=='TRANSFER' else 0)
# data['type'] = data['type'].apply(lambda x: 5 if x=='DEBIT' else x)
data['nameOrig'] = data['nameOrig'].apply(lambda x: 0 if x[0:1] == 'C' else 1)
data['nameDest'] = data['nameDest'].apply(lambda x: 0 if x[0:1] == 'C' else 1)
# Feature Engineering

data['exceeds_threshold'] = data['amount'].apply(lambda x: 1 if x > 200 else 0)
# 'step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud'

data = data[['type','amount','oldbalanceOrg','newbalanceOrig','isFraud','isFlaggedFraud']]
# data = data.drop(['amount'],axis=1)
# Split the data into features (X) and target variable (y)

X = data.drop(columns=['isFraud'])
y = data['isFraud']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
# print(data.columns)
# print(model.coef_)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

