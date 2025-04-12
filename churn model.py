import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Import Dataset
file_path = r"C:\Users\hf91_\Downloads\capstone_project\Telco_customer_churn.csv"
df = pd.read_csv(file_path)

# Data Cleaning & Preprocessing
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Total Charges'].fillna(df['Total Charges'].mean(), inplace=True)

# Feature Engineering
df['Avg_Monthly_Charges'] = df['Total Charges'] / df['Tenure Months'].replace(0, 1)
df['Tenure_Group'] = pd.cut(df['Tenure Months'], bins=[0, 12, 24, 48, 60, df['Tenure Months'].max()],
                            labels=['0-12', '13-24', '25-48', '49-60', '60+']).astype(str)

# Drop unnecessary columns
drop_cols = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
    'Lat Long', 'Latitude', 'Longitude', 'Churn Reason',
    'Churn Score', 'CLTV', 'Churn Value'
]
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Label Encoding
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    if col != 'Churn Label':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

target_le = LabelEncoder()
df['Churn Label'] = target_le.fit_transform(df['Churn Label'])

# Original dataset split
X_orig = df.drop(columns='Churn Label')
y_orig = df['Churn Label']
feature_names = X_orig.columns.tolist()

scaler = StandardScaler()
X_scaled_orig = scaler.fit_transform(X_orig)
X_scaled_orig = pd.DataFrame(X_scaled_orig, columns=feature_names)

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_scaled_orig, y_orig, test_size=0.2, random_state=42)

# Train original model
model_orig = RandomForestClassifier(random_state=42)
model_orig.fit(X_train_orig, y_train_orig)

y_pred_orig = model_orig.predict(X_test_orig)
accuracy_orig = accuracy_score(y_test_orig, y_pred_orig)
print(f"[Original Model Accuracy]: {accuracy_orig:.4f}")
print("\n[Original Random Forest Classifier Report]:")
print(classification_report(y_test_orig, y_pred_orig))
print("[Confusion Matrix]:")
print(confusion_matrix(y_test_orig, y_pred_orig))

# Upsample the minority class
df_majority = df[df['Churn Label'] == 0]
df_minority = df[df['Churn Label'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X = df_balanced.drop(columns='Churn Label')
y = df_balanced['Churn Label']

X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train upsampled model
model_upsampled = RandomForestClassifier(random_state=42)
model_upsampled.fit(X_train, y_train)

y_pred_upsampled = model_upsampled.predict(X_test)
accuracy_upsampled = accuracy_score(y_test, y_pred_upsampled)
print(f"[Upsampled Model Accuracy]: {accuracy_upsampled:.4f}")
print("\n[Upsampled Model Random Forest Classification Report]:")
print(classification_report(y_test, y_pred_upsampled))
print("[Confusion Matrix]:")
print(confusion_matrix(y_test, y_pred_upsampled))

# Hyperparameter Tuning
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False]
}

rscv = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid, n_iter=20, cv=3,
                          scoring='recall', verbose=1, n_jobs=-1, random_state=42)
rscv.fit(X_train, y_train)
best_model = rscv.best_estimator_

y_pred_best = best_model.predict(X_test)
print("[Best Parameters]:", rscv.best_params_)
print("\n[Best Model Accuracy]:", accuracy_score(y_test, y_pred_best))
print("\n[Best Model Classification Report]:")
print(classification_report(y_test, y_pred_best))
print("\n[Confusion Matrix]:")
print(confusion_matrix(y_test, y_pred_best))

# ROC & AUC Scores
y_proba_orig = model_orig.predict_proba(X_test_orig)[:, 1]
y_proba_upsampled = model_upsampled.predict_proba(X_test)[:, 1]
y_proba_tuned = best_model.predict_proba(X_test)[:, 1]

auc_orig = roc_auc_score(y_test_orig, y_proba_orig)
auc_upsampled = roc_auc_score(y_test, y_proba_upsampled)
auc_tuned = roc_auc_score(y_test, y_proba_tuned)

fpr_orig, tpr_orig, _ = roc_curve(y_test_orig, y_proba_orig)
fpr_up, tpr_up, _ = roc_curve(y_test, y_proba_upsampled)
fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_proba_tuned)

plt.figure(figsize=(8, 6))
plt.plot(fpr_orig, tpr_orig, label=f'Original (AUC = {auc_orig:.2f})')
plt.plot(fpr_up, tpr_up, label=f'Upsampled (AUC = {auc_upsampled:.2f})')
plt.plot(fpr_tuned, tpr_tuned, label=f'Tuned Upsampled (AUC = {auc_tuned:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Confusion Matrix Plot
y_pred_tuned = (y_proba_tuned >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0.5, 1.5], ['No Churn', 'Churn'])
plt.yticks([0.5, 1.5], ['No Churn', 'Churn'], rotation=0)
plt.tight_layout()
plt.show()

# Feature Importance
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(14, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette=["#A1887F", "#D3C6A6"] * 5)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Save model, encoders, scaler, and features
with open("tuned_rf_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)
