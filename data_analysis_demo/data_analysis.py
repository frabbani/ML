import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print( "Hello world!")

df = pd.read_csv("data_with_outliers.csv")
df['outlier'] = df['outlier'].astype(str).str.lower()
true_indices = df.index[df['outlier'] == 'true']

scalar = StandardScaler()
scaled_data = scalar.fit_transform(df[['value']])
iso_forest = IsolationForest(contamination=75/500)
df['anomaly'] = iso_forest.fit_predict(scaled_data)
predicted_indices = df.index[df['anomaly'] == -1]

print("***************")
print("outliers v. predicted outliers indices:")
print(true_indices.tolist())
print(predicted_indices.tolist())

set1 = set(true_indices.tolist())
set2 = set(predicted_indices.tolist())
matches = set1.intersection(set2)

len1 = len(set1)
len2 = len(set2)
total = len1
if len2 > len1:
    total = len2
print("***************")
print(f"# of matches: {len(matches)}")
print(f"total # of values: {total}")
print("***************")

df['is_outlier'] = df['anomaly'] == -1
# Plotting
plt.figure(figsize=(10,6))
plt.plot(df.index,df['value'], label="Data")
plt.scatter(df.index[df['is_outlier']], df['value'][df['is_outlier']], color='red', label='Outliers')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Anomaly Detection')
plt.legend()
plt.show()

print(f"no. of outliers detected: {df['is_outlier'].sum()}")
print("Goodbye!")
