# Online Payment Fraud Detection

This project is an end-to-end machine learning pipeline to detect fraudulent online transactions.

## Project Overview
- Preprocesses transaction data  
- Trains multiple machine learning models (Logistic Regression, Decision Tree, etc.)  
- Handles class imbalance using `class_weight="balanced"`  
- Uses probability calibration for better fraud risk scores  
- Provides a Flask web interface and API for predictions  

## How to Run
1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/Online-Payment-Fraud-Detection.git
   cd Online-Payment-Fraud-Detection
   ```
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:  
   ```bash
   python app.py
   ```
4. Open the browser at `http://127.0.0.1:8080/prediction.html` to enter transaction details and check predictions.

## Example Output
- `prediction = 0` → Not Fraud  
- `prediction = 1` → Fraud  
- Also provides fraud risk probabilities, e.g. `[0.2, 0.8]` meaning 80% chance of fraud.

## Tech Stack
- Python 3.x  
- scikit-learn  
- pandas, numpy  
- matplotlib, seaborn  
- flask  


