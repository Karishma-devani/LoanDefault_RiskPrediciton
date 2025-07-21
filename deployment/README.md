# Loan Default Prediction App (Dockerized)

This is a machine learning Flask app that predicts loan default using logistic regression.

## 🔧 Technologies Used
- Flask
- Scikit-learn
- Docker

## 🐳 How to Run with Docker

```bash
docker build -t loan-default-app .
docker run -d -p 5000:5000 loan-default-app
