import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# 1. Data Loading & Cleaning
df = pd.read_csv('data/Obesity.csv')

def preprocess_raw_data(df):
    df_copy = df.copy()
    # Rounding specific columns according to data dictionary
    round_cols = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Age']
    for col in round_cols:
        df_copy[col] = df_copy[col].round().astype(int)
    return df_copy

df_clean = preprocess_raw_data(df)

# 2. Splitting Features and Target
X = df_clean.drop('Obesity', axis=1)
y = df_clean['Obesity']

# Identifying column types
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 3. Creating Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# 4. Creating Full Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 6. Training
pipeline.fit(X_train, y_train)

# 7. Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Saving the Model
joblib.dump(pipeline, 'app/model_pipeline.pkl')
print("Model saved successfully as 'app/model_pipeline.pkl'")
