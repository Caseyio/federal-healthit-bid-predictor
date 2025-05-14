# Federal Health IT Contract Bid Predictor

This project uses real federal procurement data to predict the **expected award amount** for Health IT contracts based on contract structure, competition, and agency context.

> Built as a career-aligned case study targeting roles in **federal Health IT**, **capture management**, and **data-driven proposal strategy**.

---

## Live Applications

- 🔗 [**Award Amount Estimator**](https://ay7jcdeztbpknhyxxbn5h3.streamlit.app)  
- 🔗 [**Bid Range Predictor (Confidence Tool)**](https://federal-healthit-bid-predictor-mzxes68t2cusms5kmjuyyr.streamlit.app)

These Streamlit apps allow capture teams and analysts to explore award predictions or bid confidence intervals interactively, supporting faster and smarter PTW (Price-to-Win) decisions.

---

## Use Case

These tools help federal proposal teams, capture managers, and business analysts **estimate a competitive bid amount** using over **43,000 real contract records** (2018–2025). Just input contract attributes such as:

- ✅ NAICS code (e.g., 541512)
- ✅ Pricing type (e.g., Firm Fixed Price)
- ✅ Set-aside category
- ✅ Target agency (e.g., VA or HHS)
- ✅ Number of offers received

Get a point estimate or confidence interval for the likely award amount — fast, data-driven decision support.

---

## Apps Included

### `app.py` – Award Amount Estimator  
Returns a predicted contract award amount using a trained XGBoost regression model.

### `app_range.py` – Bid Range Confidence Tool  
Provides a **±1.71 RMSE confidence range** for bid planning based on the same model.

#### 📊 Example Output
> Estimate: \$5.2M  
> Confidence Range: \$1.2M – \$23.6M

---

## 🔍 Model Summary

- **Model**: XGBoost regressor  
- **Target**: Log-transformed total award amount  
- **Performance**: RMSE ≈ **1.71**, R² ≈ **0.35**  
- **Features**: One-hot encoded NAICS, set-aside, pricing type, and agency fields  
- **Interpretation**: SHAP values, PDP++, feature importance  
- **Deployment**: Streamlit + GitHub

---

## 📁 Repository Contents

| File / Folder             | Description                                        |
|---------------------------|----------------------------------------------------|
| `app.py`                  | Streamlit app for single award prediction          |
| `app_range.py`            | Streamlit app with confidence range output         |
| `xgb_federal_award_model.pkl` | Trained XGBoost model (log scale)            |
| `xgb_feature_columns.pkl` | List of encoded feature columns used in training   |
| `requirements.txt`        | Dependencies for running locally                   |

---

## Run Locally

To test the apps on your own machine:

```bash
# Install dependencies
pip install -r requirements.txt

# Run either app
streamlit run app.py
streamlit run app_range.py
```
---

👤 Author

Casey Ortiz
📍 Annapolis, MD
📫 kcarlos.ortiz@gmail.com
🔗 LinkedIn
🔗 GitHub
