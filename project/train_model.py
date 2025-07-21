# Complete Employee Salary Prediction Model Training Script
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")

# 1. Load data
try:
    df = pd.read_csv("dataset.csv")
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'dataset.csv' not found in current directory!")
    exit()

# 2. Display column names and basic info
print("\nColumns in dataset:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# 3. Remove unwanted columns (as per your requirements)
columns_to_remove = ['fnlwgt', 'race', 'capital-gain', 'capital-loss']
for col in columns_to_remove:
    if col in df.columns:
        df = df.drop(col, axis=1)
        print(f"Dropped '{col}' column")

# 4. Handle missing values and clean data
df = df.replace(' ?', np.nan)
df = df.replace('?', np.nan)

# Remove leading/trailing spaces from string columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip()

print(f"\nMissing values per column:\n{df.isnull().sum()}")

# Drop rows with missing values
initial_rows = len(df)
df = df.dropna()
print(f"Dropped {initial_rows - len(df)} rows with missing values. Remaining: {len(df)}")

# 5. Simplify marital status to 3 categories (as per your requirements)
if 'marital-status' in df.columns:
    def simplify_marital_status(status):
        if 'Married' in str(status):
            return 'Married'
        elif 'Never-married' in str(status):
            return 'Never-married'
        else:
            return 'Divorced/Separated/Widowed'
    
    df['marital-status'] = df['marital-status'].apply(simplify_marital_status)
    print(f"Simplified marital status categories: {df['marital-status'].unique()}")

# 6. Check unique values in income column
print(f"\nUnique values in income column: {df['income'].unique()}")

# 7. Remove outliers from numerical columns
def remove_outliers(df, cols):
    """Remove outliers using IQR method"""
    for col in cols:
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before_count = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            print(f"Removed {before_count - len(df)} outliers from '{col}'. Remaining: {len(df)}")
    return df

# Remove outliers from remaining numerical columns
numerical_cols = ['age', 'hours-per-week', 'educational-num']
existing_numerical_cols = [col for col in numerical_cols if col in df.columns]
print(f"\nRemoving outliers from: {existing_numerical_cols}")
df = remove_outliers(df, existing_numerical_cols)

print(f"Final dataset shape after cleaning: {df.shape}")

# 8. Prepare features and target
X = df.drop('income', axis=1)
y = df['income']

print(f"\nTarget variable distribution:")
print(y.value_counts())

# 9. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# 10. Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols),
    ('num', 'passthrough', numerical_cols)
])

# 11. Train CLASSIFICATION model
print("\n=== Training Classification Model ===")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

clf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
clf_model.fit(X_train, y_train)

y_pred = clf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 12. Train REGRESSION model for salary estimation
print("\n=== Training Regression Model for Salary Estimation ===")

def income_to_salary(income_str):
    """Convert income category to estimated salary value"""
    if '<=50K' in str(income_str) or ' <=50K' in str(income_str):
        return 35000  # Average for <=50K category
    else:
        return 75000  # Average for >50K category

y_salary = df['income'].apply(income_to_salary)
print(f"Salary target distribution:")
print(y_salary.value_counts())

reg_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_salary, test_size=0.2, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)

y_pred_reg = reg_model.predict(X_test_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"Regression MAE: {mae:.2f}")
print(f"Regression RMSE: {rmse:.2f}")

# 13. Save models and metadata
print("\n=== Saving Models ===")

joblib.dump(clf_model, "salary_classification_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(reg_model, "salary_regression_model.pkl")

column_info = {
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'all_columns': list(X.columns)
}
joblib.dump(column_info, "column_info.pkl")

print("Models saved successfully!")
print("\nFiles created:")
print("- salary_classification_model.pkl (predicts income category)")
print("- salary_regression_model.pkl (estimates salary value)")
print("- label_encoder.pkl (for decoding predictions)")
print("- column_info.pkl (column metadata)")

print("\n=== Model Training Complete ===")
print("Columns removed: fnlwgt, race, capital-gain, capital-loss")
print("Marital status simplified to 3 categories")
print("You can now run the Streamlit app with: streamlit run app.py")
