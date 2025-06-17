# Install necessary libraries
!pip install pandas numpy scikit-learn matplotlib seaborn sqlite3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic ophthalmology dataset with 500 patients
print("Generating synthetic ophthalmology dataset with 500 patients...")

# Generate realistic patient IDs
patient_ids = [f"P{10000 + i}" for i in range(1, 501)]

# Generate synthetic data with realistic distributions
data = {
    'patient_id': patient_ids,
    'age': np.random.normal(60, 15, 500).astype(int),
    'intraocular_pressure': np.clip(np.random.normal(16, 4, 500), 8, 40),
    'visual_acuity': np.clip(np.random.normal(0.7, 0.2, 500), 0.1, 1.0),
    'retinal_thickness': np.clip(np.random.normal(280, 40, 500), 150, 400),
    'diabetes': np.random.choice([0, 1], size=500, p=[0.7, 0.3]),
    'hypertension': np.random.choice([0, 1], size=500, p=[0.6, 0.4]),
    'family_history': np.random.choice([0, 1], size=500, p=[0.8, 0.2]),
    'smoking': np.random.choice([0, 1], size=500, p=[0.75, 0.25]),
}

# Create DataFrame
df = pd.DataFrame(data)

# Add disease progression with realistic patterns
df['disease_progression'] = np.where(
    (df['intraocular_pressure'] > 21) | (df['retinal_thickness'] < 220),
    'Progressed',
    'Stable'
)

# Add diagnosis with realistic patterns
def assign_diagnosis(row):
    if row['intraocular_pressure'] > 21 and row['retinal_thickness'] > 250:
        return 'Glaucoma'
    elif row['retinal_thickness'] < 220 and row['visual_acuity'] < 0.6:
        return 'AMD'
    elif row['diabetes'] == 1 and row['retinal_thickness'] < 250:
        return 'Diabetic Retinopathy'
    else:
        return 'Normal'

df['diagnosis'] = df.apply(assign_diagnosis, axis=1)

# Step 2: Exploratory Data Analysis (EDA)
print("\nPerforming exploratory data analysis...")

# Basic dataset info
print(f"\nDataset shape: {df.shape}")
print("\nFirst 5 records:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Visualize distribution of key metrics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(df['intraocular_pressure'], kde=True, color='red')
plt.title('Distribution of Intraocular Pressure')
plt.xlabel('Intraocular Pressure (mmHg)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
sns.histplot(df['retinal_thickness'], kde=True, color='green')
plt.title('Distribution of Retinal Thickness')
plt.xlabel('Retinal Thickness (μm)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
sns.histplot(df['visual_acuity'], kde=True, color='purple')
plt.title('Distribution of Visual Acuity')
plt.xlabel('Visual Acuity')
plt.ylabel('Frequency')

plt.subplot(2, 2, 4)
sns.histplot(x='diagnosis', data=df, palette='viridis')
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('ophthalmology_eda.png')
plt.show()

# Visualize correlation heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

# Step 3: Predictive Modeling
print("\nTraining predictive model for diagnosis...")

# Prepare data for modeling
X = df[['age', 'intraocular_pressure', 'visual_acuity', 'retinal_thickness',
        'diabetes', 'hypertension', 'family_history', 'smoking']]
y = df['diagnosis']

# Encode categorical target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy * 100:.2f}%')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importances
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Step 4: Population trends and treatment outcomes
print("\nAnalyzing population trends and treatment outcomes...")

# Age group analysis
age_groups = pd.cut(df['age'], bins=[20, 40, 50, 60, 70, 90],
                    labels=['20-40', '40-50', '50-60', '60-70', '70+'])
df['age_group'] = age_groups

# Diagnosis by age group
plt.figure(figsize=(10, 6))
sns.countplot(x='age_group', hue='diagnosis', data=df, palette='viridis')
plt.title('Diagnosis Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.legend(title='Diagnosis')
plt.savefig('diagnosis_by_age.png')
plt.show()

# Treatment outcomes analysis
treatment_outcomes = df.groupby(['diagnosis', 'disease_progression']).size().unstack()
print("\nTreatment Outcomes by Diagnosis:")
print(treatment_outcomes)

# Progression rates
progression_rates = df.groupby('diagnosis')['disease_progression'].value_counts(normalize=True).unstack()
print("\nDisease Progression Rates:")
print(progression_rates)

# Step 5: Store insights in a structured database
print("\nStoring data and insights in SQLite database...")

# Connect to SQLite database
conn = sqlite3.connect('ophthalmology_insights.db')

# Create tables
df.to_sql('patient_data', conn, if_exists='replace', index=False)
feature_importances.to_sql('feature_importances', conn, if_exists='replace', index=False)

# Create summary tables
# Diagnosis summary
diagnosis_summary = df['diagnosis'].value_counts().reset_index()
diagnosis_summary.columns = ['diagnosis', 'count']
diagnosis_summary.to_sql('diagnosis_summary', conn, if_exists='replace', index=False)

# Age group summary
age_summary = df.groupby('age_group').agg(
    avg_iop=('intraocular_pressure', 'mean'),
    avg_retinal_thickness=('retinal_thickness', 'mean'),
    progression_rate=('disease_progression', lambda x: (x == 'Progressed').mean())
).reset_index()
age_summary.to_sql('age_group_summary', conn, if_exists='replace', index=False)

# Commit and close
conn.commit()
conn.close()

print("✅ Data and insights stored in 'ophthalmology_insights.db'")

# Step 6: Query the database to verify
print("\nQuerying database to verify data...")

conn = sqlite3.connect('ophthalmology_insights.db')

# Query patient data
patient_data = pd.read_sql_query("SELECT * FROM patient_data LIMIT 5;", conn)
print("\nSample Patient Data:")
print(patient_data)

# Query feature importances
feature_importances_db = pd.read_sql_query("SELECT * FROM feature_importances;", conn)
print("\nFeature Importances:")
print(feature_importances_db)

# Query diagnosis summary
diagnosis_summary_db = pd.read_sql_query("SELECT * FROM diagnosis_summary;", conn)
print("\nDiagnosis Summary:")
print(diagnosis_summary_db)

# Close connection
conn.close()

print("\nAnalysis complete! Key visualizations saved as PNG files.")