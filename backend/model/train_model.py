import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

print("Loading data...")
data_path = "loan_data.csv"
df = pd.read_csv(data_path)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

print("Encoding categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(f"After encoding: {df.shape[1]} columns")

# Feature Engineering
print("Performing feature engineering...")
df['credit_utilization'] = df['revol.bal'] / ((df['revol.bal'] * 1.2) + 1)
df['adjusted_dti'] = df['dti'] / (df['installment'] + 1)
df['days_sales_outstanding'] = df['days.with.cr.line'] / (df['fico'] + 1)

# This Handles Missing Values
print("Filling missing values...")
df.fillna(df.median(), inplace=True)

X = df.drop(columns=['not.fully.paid'])  # Features
y = df['not.fully.paid']  # Target variable
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Standardizing the Features
print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# This Handles Class Imbalance with SMOTE
print("Applying SMOTE to balance classes...")
sm = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(f"After SMOTE: {X_train.shape[0]} training samples")

# Hyperparameter Tuning Function
def tune_model(model, param_grid):
    print(f"Tuning {model.__class__.__name__} with RandomizedSearchCV...")
    grid = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, random_state=42, n_iter=30, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid.best_params_}")
    return grid.best_estimator_

# Logistic Regression
print("Starting Logistic Regression training...")
log_reg_params = {
    'C': np.logspace(-3, 3, 10),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
log_reg = tune_model(LogisticRegression(), log_reg_params)

# Decision Tree
print("Starting Decision Tree training...")
dt_params = {
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}
dt = tune_model(DecisionTreeClassifier(), dt_params)

# Random Forest
print("Starting Random Forest training...")
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'bootstrap': [True, False]
}
rf = tune_model(RandomForestClassifier(), rf_params)

# Stacking Model
print("Starting Stacking Classifier training...")
stacking_model = StackingClassifier(
    estimators=[('lr', log_reg), ('dt', dt), ('rf', rf)],
    final_estimator=LogisticRegression(),
    passthrough=True,
    verbose=1
)
stacking_model.fit(X_train, y_train)
print("Stacking Classifier training completed.")

# Evaluating Models
models = {
    'Logistic Regression': log_reg,
    'Decision Tree': dt,
    'Random Forest': rf,
    'Stacking Model': stacking_model
}

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)
    print(f"🔹 {name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"🔹 ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
    print(classification_report(y_test, y_pred))

# Save the Best Model
print("Determining and saving the best model...")
best_model = max(models, key=lambda name: accuracy_score(y_test, models[name].predict(X_test)))
joblib.dump(models[best_model], 'best_credit_risk_model.pkl', compress=3)
print(f"Best model ({best_model}) saved as 'best_credit_risk_model.pkl'")

# Feature Importance (Random Forest)
print("Generating feature importance plot for Random Forest...")
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances['Feature'])
plt.title('Feature Importance - Random Forest')
plt.savefig('feature_importance_rf.png')
print("Feature importance plot saved as 'feature_importance_rf.png'")

# SHAP Explainability
print("Generating SHAP summary plots...")
for model in [log_reg, dt, rf]:
    print(f"Computing SHAP values for {model.__class__.__name__}...")
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    print(f"SHAP values computed for {model.__class__.__name__}. Generating plot...")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary Plot - {model.__class__.__name__}")
    plt.savefig(f'shap_summary_{model.__class__.__name__}.png')
    plt.close()
    print(f"SHAP summary plot saved as 'shap_summary_{model.__class__.__name__}.png'")

print("Training script completed!")