# Credit Risk Prediction Project

## Overview

##Problem Statement: Predict the probability of a borrower defaulting on a loan.

This project builds a machine learning model to predict credit risk for loan applicants based on financial features. The system uses a dataset (`loan_data.csv`) to train a stacking classifier combining Logistic Regression, Decision Tree, and Random Forest models. The trained model is deployed via a Flask backend and a simple HTML frontend, allowing users to input financial details and receive a risk prediction (Low Risk or High Risk) along with a probability score.

The project includes feature engineering, hyperparameter tuning, model explainability (via SHAP), and a rule-based override to ensure high debt-to-income ratios (DTI) are flagged appropriately.

---

## Technologies Used

- **Python 3.9+**: Core programming language for data processing, model training, and backend development.
- **Pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computations.
- **Scikit-learn**: Machine learning models (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, StackingClassifier), preprocessing (StandardScaler), and evaluation metrics.
- **Imbalanced-learn (imblearn)**: SMOTE for handling class imbalance.
- **SHAP**: Model explainability and feature importance analysis.
- **Matplotlib & Seaborn**: Visualization of feature importance and SHAP plots.
- **Joblib**: Model and scaler serialization.
- **Flask**: Backend API for serving predictions.
- **Flask-CORS**: Handling cross-origin requests.
- **HTML/CSS/JavaScript**: Frontend interface for user input and displaying predictions.
- **Tailwind CSS**: Styling the frontend form.

---

## Project Structure

- **loan_data.csv**: Dataset containing loan applicant data (e.g., `fico`, `dti`, `revol.bal`, `not.fully.paid`).
- **train_credit_risk.py**: Script to train the model, perform feature engineering, tune hyperparameters, and save the best model.
- **app.py**: Flask backend to serve predictions.
- **index.html**: Frontend interface for user input and results.
- **scaler.pkl**: Saved StandardScaler for feature scaling.
- **best_credit_risk_model.pkl**: Saved stacking classifier model.
- **feature_importance_rf.png**: Feature importance plot for Random Forest.
- **shap_summary_LogisticRegression.png**: SHAP summary plot for Logistic Regression.
- **images/**: Directory containing screenshots of the frontend and plots.

---

## How to Run

### Prerequisites
- Python 3.9+
- Install dependencies:
  ```bash
  pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn flask flask-cors joblib
