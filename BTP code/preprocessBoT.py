import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from google.colab import files

# Upload dataset from local system
uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Drop empty columns
empty_columns = ['smac', 'dmac', 'soui', 'doui', 'sco', 'dco']
df.drop(columns=empty_columns, inplace=True)

# Drop non-informative columns
drop_columns = ['pkSeqID', 'ltime', 'seq']  # 'pkSeqID' is just an ID, 'ltime' & 'seq' may not be useful
df.drop(columns=drop_columns, inplace=True)

# Handle missing values (fill with 0 or drop rows)
df.fillna(0, inplace=True)  # Replacing NaN with 0, can be changed if necessary

# Encode categorical features
label_encoders = {}
categorical_columns = ['proto', 'flgs', 'category', 'subcategory', 'state']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convert to string first if needed
    label_encoders[col] = le  # Save encoder if needed for inverse transform

# Normalize numerical features
scaler = MinMaxScaler()
numerical_columns = [col for col in df.columns if col not in ['attack']]
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Save preprocessed dataset
df.to_csv("bot_iot_preprocessed.csv", index=False)

print("Preprocessing complete. Processed file saved as bot_iot_preprocessed.csv")

# Data Visualization
plt.figure(figsize=(12, 6))
sns.countplot(x='attack', data=df, palette='coolwarm')
plt.title('Attack Distribution')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# Sample pairplot of selected features
selected_features = ['pkts', 'bytes', 'sbytes', 'dbytes', 'rate', 'attack']
sns.pairplot(df[selected_features], hue='attack', palette='coolwarm')
plt.show()
