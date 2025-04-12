# Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Define the local file path
file_path = r"C:\Users\hf91_\Downloads\capstone_project\Telco_customer_churn.csv"

# Check if the file exists before loading
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("✅ File loaded successfully.")
else:

    print(f"❌ File not found at: {file_path}")
    print("Please check the file path and ensure the CSV file exists.")

# Explore Dataset
print(df)

# Check data types
df.info()

# -------------------------------
# Data Cleaning & Preprocessing
# -------------------------------
# Convert 'Total Charges' to numeric
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Fill missing 'Total Charges' with the mean
df['Total Charges'].fillna(df['Total Charges'].mean(), inplace=True)

# -------------------------------
# Data Visualization
# -------------------------------
# Set style
sns.set(style="whitegrid")

# Define custom palette
churn_palette = {"Yes": "#b08968", "No": "#b7b7a4"}

# 1. Churn Distribution
churn_counts = df['Churn Label'].value_counts()

# Create a pie chart for churn counts
plt.figure(figsize=(10, 8))
plt.pie(
    churn_counts,
    labels=churn_counts.index,
    autopct='%1.1f%%',
    colors=[churn_palette[label] for label in churn_counts.index]
)
plt.title('Churn Distribution')
plt.show()

# 2. Monthly Charges vs Churn
# Group by Churn Label and calculate the mean of Monthly Charges
monthly_charges_by_churn = df.groupby('Churn Label')['Monthly Charges'].mean()

# Create a pie chart for Monthly Charges vs Churn
plt.figure(figsize=(10, 8))
plt.pie(
    monthly_charges_by_churn,
    labels=monthly_charges_by_churn.index,
    autopct='%1.1f%%',
    colors=[churn_palette[label] for label in monthly_charges_by_churn.index],
    startangle=90
)
plt.title('Monthly Charges vs Churn')
plt.show()

# 3. Tenure Distribution by Churn Status
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Tenure Months', hue='Churn Label', kde=True, multiple='stack', element='step',
             palette=churn_palette)
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Tenure (Months)')
plt.ylabel('Customer Count')
plt.show()

# 4. Contract Type vs Churn
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Contract', hue='Churn Label', palette=churn_palette)
plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.xticks(rotation=20)
plt.legend(title='Churn')
plt.show()

# 5. Internet Service vs Churn
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Internet Service', hue='Churn Label', palette=churn_palette)
plt.title('Churn by Internet Service Type')
plt.xlabel('Internet Service')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.show()

# 6. Payment Method vs Churn
plt.figure(figsize=(18,8))
sns.countplot(data=df, y='Payment Method', hue='Churn Label', palette=churn_palette)
plt.title('Churn by Payment Method')
plt.xlabel('Count')
plt.ylabel('Payment Method')
plt.legend(title='Churn')
plt.show()

# ----- Correlation Heatmap -----
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()