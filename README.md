# 📊 Federal Health IT Contract Bid Predictor

This project uses real federal procurement data to predict the **expected award amount** for Health IT contracts based on contract structure, competition, and agency context.

> Built as a career-aligned data science case study targeting roles in **federal health IT**, **capture management**, and **data-driven proposal strategy**.

---

## 📈 Health IT Bid Confidence Tool (`app_range.py`)

This companion app provides a **confidence range** instead of a single prediction. It helps capture managers and proposal strategists better understand the **award uncertainty** and plan more effectively.

🧠 Based on the model’s RMSE of ~1.71 (log scale), most predictions fall within a **±4.5x range**.

### ✨ Example Output

---

## 🚀 Live App

🔗 [Launch Award Amount Streamlit App]([https://your-app-link.streamlit.app](https://ay7jcdeztbpknhyxxbn5h3.streamlit.app))  
🔗 [Launch Bid Range Predictor Streamlit App]([https://your-app-link.streamlit.app](https://ay7jcdeztbpknhyxxbn5h3.streamlit.app))  

---

## 🎯 Use Case

These apps helps proposal teams, capture managers, or analysts **estimate a competitive bid amount** for federal contracts in the Health IT space. Input key contract features such as:

- NAICS code (e.g., 541512)
- Pricing type (e.g., Firm Fixed Price)
- Set-aside structure
- Agency
- Number of offers expected

…and get a predicted award amount or bid range based on over **43,000 real contract records** from 2018–2025.

---

## 🧠 Model Details

- Model: `XGBoost` regressor (final RMSE ≈ **1.71**, R² ≈ **0.35**)
- Features: One-hot encoded NAICS, set-aside, pricing type, and agency fields
- Target: Log-transformed total award amount
- Interpretation: SHAP values, PDP++, and feature importance tracking
- Deployment: Streamlit + GitHub

---

## 📁 Files

- `app.py`: Streamlit web app for award amount
- `app_range.py`: Streamlit web app for bid range
- `xgb_federal_award_model.pkl`: Trained XGBoost model
- `xgb_feature_columns.pkl`: List of encoded features used in training

---

## 🛠️ Setup Locally

```bash
pip install -r requirements.txt
streamlit run app.py

pip install -r requirements.txt
streamlit run app_range.py

## 👤 Author

**Casey Ortiz**  
📍 Annapolis, MD  
📫 kcarlos.ortiz@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/kco1) | [GitHub](https://github.com/caseio)
