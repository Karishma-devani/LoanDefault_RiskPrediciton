import pickle
import traceback
import joblib
from flask import Flask, render_template, request, jsonify
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# Load models
model = joblib.load("model_top10_smote.pkl")

#Saving Flask app
app = Flask(__name__)

#Selected Features
ALLOWED_INPUT_FEATURES = [
    'lp_grossprincipalloss',
    'lp_netprincipalloss',
    'loancurrentdaysdelinquent',
    'lp_collectionfees',
    'delinquency_risk_score',
    'lp_nonprincipalrecoverypayments',
    'monthlyloanpayment',
    'investors',
    'loan_to_income',
    'lp_customerpayments'
]

#Function to Create a dataframe with the user input value 
def preprocess_input(user_input: dict) -> pd.DataFrame:
  
    # Initialize with 0s
    full_input = {feature: 0 for feature in ALLOWED_INPUT_FEATURES}
    
    # Fill with actual values where provided
    for key in ALLOWED_INPUT_FEATURES:
        if key in user_input:
            full_input[key] = user_input[key]
        else:
            raise ValueError(f"Missing required feature: {key}")
    
    df = pd.DataFrame([full_input])
    df = df[ALLOWED_INPUT_FEATURES]  # Ensure correct column order
    return df

# Route to show the input form
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Renders form

# Route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather form data
        user_input = {}
        for feature in ALLOWED_INPUT_FEATURES:
            value = request.form.get(feature)
            if value is None:
                return f"Missing: {feature}", 400
            user_input[feature] = float(value)  # or int(value) if needed
        processed_df = preprocess_input(user_input)
        
        # Predict
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0][1]
        
        return render_template('result.html', prediction=prediction)
	
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

