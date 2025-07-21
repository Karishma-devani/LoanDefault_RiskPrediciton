import requests

url = "http://127.0.0.1:5000/predict"

sample_input = {
    "lp_grossprincipalloss": 300.0,              # Some loss due to default
    "lp_netprincipalloss": 250.0,                # Net loss after recoveries
    "loancurrentdaysdelinquent": 15,             # Days overdue
    "lp_collectionfees": 25.0,                   # Fees collected
    "delinquency_risk_score": 640.0,             # Risk score
    "lp_nonprincipalrecoverypayments": 50.0,     # Recovered non-principal
    "monthlyloanpayment": 350.0,                 # Monthly EMI
    "investors": 12,                             # Number of investors
    "loan_to_income": 0.35,                      # 35% loan-to-income
    "lp_customerpayments": 5000.0                # Payments made
}

response = requests.post(url, json=sample_input)
print("Response:", response.json())