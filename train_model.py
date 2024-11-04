import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('data/large_hospital_cost_prediction_dataset (1).csv')

# Preprocessing
# Check for duplicates
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()

# Fill missing values in Comorbidities
df['Comorbidities'] = df['Comorbidities'].fillna('None')

# For other numeric columns, fill NaN with median
df['Length_of_Stay'] = df['Length_of_Stay'].fillna(df['Length_of_Stay'].median())  
df['Procedure_Cost'] = df['Procedure_Cost'].fillna(df['Procedure_Cost'].median())
df['Medication_Cost'] = df['Medication_Cost'].fillna(df['Medication_Cost'].median())

# Clean the DataFrame by dropping unnecessary columns
df = df.drop(columns=['Patient_ID'])  # Drop Patient_ID as it's not needed for training

# Convert categorical variables to one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Admission_Type', 'Diagnosis_Code', 
                                   'Treatment_Type', 'Insurance_Type', 
                                   'Discharge_Status', 'Comorbidities'], drop_first=True)
# Splitting the data into features and target variable
X = df.drop('Total_Cost', axis=1)  # Features
y = df['Total_Cost']  # Target variable

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Save the model
joblib.dump(model, 'models/hospital_cost_model.pkl')
print("Model saved as 'models/hospital_cost_model.pkl'")

