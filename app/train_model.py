import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Data Loading & Cleaning
df = pd.read_csv('/content/drive/MyDrive/GitHub/obesity-risk-prediction/data/Obesity.csv')

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

# 3. Building the Pipeline
# Preprocessing for numerical: Scaling
# Preprocessing for categorical: One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Full pipeline: Preprocessing + Classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 4. Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

# Predictions
y_pred = model_pipeline.predict(X_test)

# Metrics
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 5. Exporting for Deployment (Streamlit)
# This saves the entire pipeline (scaler + encoder + model)
joblib.dump(model_pipeline, '/content/drive/MyDrive/GitHub/obesity-risk-prediction/app/model_pipeline.pkl')
print("Model saved successfully as 'app/model_pipeline.pkl'")