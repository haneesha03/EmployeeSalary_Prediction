# Employee Salary Prediction — Streamlit app

This repository contains a Jupyter notebook and a small Streamlit app that trains a Linear Regression model to predict employee salary from basic features.

Files added:
- `streamlit_app.py` — Streamlit application that loads the CSV, trains the preprocessing pipeline (LabelEncoders + StandardScalers) and a LinearRegression model, and exposes a UI to predict salary from inputs.
- `requirements.txt` — Python packages required to run the app.

How to run (Windows PowerShell):

1. (Optional) Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app from the `project2` folder where the dataset `Dataset09-Employee-salary-prediction.csv` is located:

```powershell
streamlit run streamlit_app.py
```

Notes:
- The app will look for `Dataset09-Employee-salary-prediction.csv` in the same folder by default. You can also upload a CSV via the sidebar; it must have the columns: `Age, Gender, Degree, Job_Title, Experience_years, Salary`.
- Right now the app trains the model on startup and uses scikit-learn's LabelEncoder fit to the training data. For production, consider saving a trained artifact and loading it in the app instead of retraining at each start.

Next steps (optional):
- Persist the trained model and scalers to disk (joblib/pickle) in the notebook and update the app to load them.
- Add more robust handling of unseen categorical values (e.g., map to "Other" or use OneHot/OrdinalEncoder pipelines).
