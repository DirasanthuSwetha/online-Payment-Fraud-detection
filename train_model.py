import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load full dataset
df = pd.read_csv(r"F:\4-2\4-2\smote_resampled_fraud_dataset_clean.csv")

# Drop irrelevant columns
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)

# Encode 'type' column
df['type'] = df['type'].astype('category').cat.codes

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Select only 11 lakh rows for speed
df_subset = df.iloc[:1100000]

# Split features and labels
X = df_subset.drop('isFraud', axis=1)
y = df_subset['isFraud']

# Manual split: 10 lakh for training, 1 lakh for testing

X_train = X.iloc[:1000000]
y_train = y.iloc[:1000000]
X_test = X.iloc[1000000:1100000]
y_test = y.iloc[1000000:1100000]


# Save train/test files
X_train.to_csv("X_train1.csv", index=False)
y_train.to_csv("y_train1.csv", index=False)
X_test.to_csv("X_test1.csv", index=False)
y_test.to_csv("y_test1.csv", index=False)

print("Files saved.")

# Scale training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Accuracy
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
print("Training Accuracy:", train_acc)

# Save model and scaler
with open("model1.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler1.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved as model1.pkl and scaler1.pkl")