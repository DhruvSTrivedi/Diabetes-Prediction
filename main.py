# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('diabetes.csv')
print("✅ Dataset loaded successfully!")

# Check class distribution
print("\nClass Distribution (Outcome Counts):")
print(df['Outcome'].value_counts())

# Compute mean values for both diabetic and non-diabetic cases
print("\nFeature Means by Outcome:")
print(df.groupby('Outcome').mean())

# Handle missing/zero values in critical columns
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

# Separate features (X) and target (y)
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values  # Labels

# Standardize features for better model performance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets (Stratified to maintain class balance)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Display dataset shapes
print("\nDataset Shapes:")
print(f"Total Samples: {X.shape}, Training Samples: {X_train.shape}, Testing Samples: {X_test.shape}")

# Import Machine Learning models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Define ML Pipelines (Scaling + PCA + Model)
pipelines = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('lr', LogisticRegression())]),
    'Support Vector Classifier': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('svc', SVC(kernel='linear'))]),
    'Decision Tree': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('dt', DecisionTreeClassifier())]),
    'Random Forest': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('rf', RandomForestClassifier(n_estimators=200))]),
    'K-Means': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('km', KMeans(n_clusters=2, random_state=0))]),
    'Naive Bayes': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('gnb', GaussianNB())]),
    'XGBoost': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('xgb', XGBClassifier())]),
    'K-Nearest Neighbors': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('knb', KNeighborsClassifier())])
}

# Train all models and store results
print("\n🔹 Model Performance:")
for name, model in pipelines.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name} Test Accuracy: {acc:.4f}")

# Select best model based on accuracy
best_model_name = max(pipelines, key=lambda k: pipelines[k].score(X_test, y_test))
best_model_acc = pipelines[best_model_name].score(X_test, y_test)
print(f"\n🏆 Best Model: {best_model_name} with Accuracy: {best_model_acc:.4f}")

# Train Logistic Regression for detailed metrics
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Import evaluation metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Display classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Compute accuracy scores
train_acc = accuracy_score(lr_model.predict(X_train), y_train) * 100
test_acc = accuracy_score(y_pred, y_test) * 100

print(f"\n✅ Training Accuracy: {train_acc:.2f}%")
print(f"✅ Testing Accuracy: {test_acc:.2f}%")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------------- STACKING MODEL ---------------- #
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# Define improved LightGBM parameters
lgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 10,  # Increase depth
    'num_leaves': 64,  # More leaves per tree
    'min_child_samples': 5,  # Allow smaller splits
    'min_gain_to_split': 0.01  # Ensure meaningful splits
}

# Define base models for stacking
stacking_base_models = [
    ('catboost', CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0)),
    ('lightgbm', LGBMClassifier(**lgbm_params)),  # Updated parameters
    ('xgboost', XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6))
]

# Define meta-model (Logistic Regression)
meta_model = LogisticRegression(max_iter=500)

# Stacking Classifier
stacking_model = StackingClassifier(estimators=stacking_base_models, final_estimator=meta_model, passthrough=True)

# Train Stacking Model
stacking_model.fit(X_train, y_train)

# Evaluate Stacking Model
stacking_preds = stacking_model.predict(X_test)
stacking_acc = accuracy_score(y_test, stacking_preds)

print(f"\n🚀 Updated Stacking Model Accuracy: {stacking_acc:.4f}")
