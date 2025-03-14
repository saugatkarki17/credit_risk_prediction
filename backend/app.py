from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import math

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

# Load trained model and scaler
try:
    model = joblib.load('model/models/best_credit_risk_model.pkl')
    scaler = joblib.load('model/models/scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # This extracts raw inputs
        required_fields = ['total_debt', 'annual_income', 'fico', 'days_with_cr_line', 'purpose']
        for field in required_fields:
            if field not in data or data[field] is None:
                return jsonify({"error": f"Missing field: {field}"}), 400

        total_debt = float(data['total_debt'])
        annual_income = float(data['annual_income'])
        fico = float(data['fico'])
        days_with_cr_line = float(data['days_with_cr_line'])
        purpose = data['purpose']

        # Additional features
        dti = total_debt / annual_income if annual_income > 0 else 0.1  # Raw DTI
        revol_bal = total_debt * 0.3  # 30% of debt as revolving balance
        installment = total_debt / 36  # 3-year loan term
        log_annual_inc = math.log(annual_income + 1)  # Log transform income
        revol_util = min(revol_bal / (fico * 100), 1.0)  # Based on FICO

        credit_policy = 1.0
        int_rate = 0.10 if dti < 0.4 else (0.15 if dti < 1.0 else 0.20)  # Higher rates for higher DTI
        inq_last_6mths = 0.0 if dti < 0.4 else (1.0 if dti < 1.0 else 2.0)  # More inquiries for higher DTI
        delinq_2yrs = 0.0
        pub_rec = 0.0

        # Feature Engineering
        credit_utilization = revol_bal / ((revol_bal * 1.2) + 1)
        adjusted_dti = dti / (installment + 1)  # Keep for consistency
        days_sales_outstanding = days_with_cr_line / (fico + 1)

        purpose_categories = ['credit_card', 'debt_consolidation', 'educational', 
                              'home_improvement', 'major_purchase', 'small_business']
        purpose_dummies = [1 if purpose == cat else 0 for cat in purpose_categories]

        # Creating feature array (21 features)
        features = np.array([
            credit_policy, int_rate, installment, log_annual_inc, dti, fico, 
            days_with_cr_line, revol_bal, revol_util, inq_last_6mths, delinq_2yrs, 
            pub_rec, *purpose_dummies, credit_utilization, adjusted_dti, days_sales_outstanding
        ]).reshape(1, -1)

        feature_names = [
            "credit_policy", "int_rate", "installment", "log_annual_inc", "dti", "fico", 
            "days_with_cr_line", "revol_bal", "revol_util", "inq_last_6mths", "delinq_2yrs", 
            "pub_rec", "purpose_credit_card", "purpose_debt_consolidation", "purpose_educational",
            "purpose_home_improvement", "purpose_major_purchase", "purpose_small_business",
            "credit_utilization", "adjusted_dti", "days_sales_outstanding"
        ]
        print("Feature array before scaling:")
        for name, value in zip(feature_names, features[0]):
            print(f"  {name}: {value}")

        # Verifying feature count to match with the model feature number
        if features.shape[1] != 21:
            return jsonify({"error": f"Expected 21 features, got {features.shape[1]}"}), 500

        # Scaling the features
        features_scaled = scaler.transform(features)

        # Prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        if dti > 0.4:  # Common lending threshold
            probability = max(probability, 0.6)  # Boost probability of High Risk
            prediction = 1 if probability > 0.5 else 0  # Adjust prediction accordingly

        return jsonify({
            'prediction': int(prediction),
            'probability': round(probability, 4),
            'features_engineered': {
                'credit_utilization': round(credit_utilization, 4),
                'adjusted_dti': round(adjusted_dti, 4),
                'days_sales_outstanding': round(days_sales_outstanding, 4)
            },
            'raw_dti': round(dti, 4)  # For debugging
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid input value: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)