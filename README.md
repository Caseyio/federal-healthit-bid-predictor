# 📊 Federal Health IT Contract Bid Predictor

This project uses real federal procurement data to predict the **expected award amount** for Health IT contracts based on contract structure, competition, and agency context.

> Built as a career-aligned data science case study targeting roles in **federal health IT**, **capture management**, and **data-driven proposal strategy**.

---

## 🚀 Live App

🔗 [Launch Streamlit App](https://your-app-link.streamlit.app)  
*(Replace with your actual link after deployment)*

---

## 🎯 Use Case

The app helps proposal teams, capture managers, or analysts **estimate a competitive bid amount** for federal contracts in the Health IT space. Input key contract features such as:

- NAICS code (e.g., 541512)
- Pricing type (e.g., Firm Fixed Price)
- Set-aside structure
- Agency
- Number of offers expected

…and get a predicted award amount based on over **43,000 real contract records** from 2018–2025.

---

## 🧠 Model Details

- Model: `XGBoost` regressor (final RMSE ≈ **1.71**, R² ≈ **0.35**)
- Features: One-hot encoded NAICS, set-aside, pricing type, and agency fields
- Target: Log-transformed total award amount
- Interpretation: SHAP values, PDP++, and feature importance tracking
- Deployment: Streamlit + GitHub

---

## 📁 Files

- `app.py`: Streamlit web app
- `xgb_federal_award_model.pkl`: Trained XGBoost model
- `xgb_feature_columns.pkl`: List of encoded features used in training

---

## 🛠️ Setup Locally

```bash
pip install -r requirements.txt
streamlit run app.py

## 👤 Author

**Casey Ortiz**  
📍 Annapolis, MD  
📫 kcarlos.ortiz@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/kco1) | [GitHub](https://github.com/caseio)
