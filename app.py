import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure expected column names
    df.columns = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_years', 'Salary']
    df = df.drop_duplicates().dropna()
    return df


@st.cache_resource
def train_model(df: pd.DataFrame) -> dict:
    # Fit encoders
    le_gender = LabelEncoder()
    le_degree = LabelEncoder()
    le_job = LabelEncoder()

    df['Gender_Encode'] = le_gender.fit_transform(df['Gender'])
    df['Degree_Encode'] = le_degree.fit_transform(df['Degree'])
    df['Job_Title_Encode'] = le_job.fit_transform(df['Job_Title'])

    # Fit scalers
    scaler_age = StandardScaler()
    scaler_exp = StandardScaler()
    df['Age_scaled'] = scaler_age.fit_transform(df[['Age']])
    df['Experience_years_scaled'] = scaler_exp.fit_transform(df[['Experience_years']])

    X = df[['Age_scaled', 'Gender_Encode', 'Degree_Encode', 'Job_Title_Encode', 'Experience_years_scaled']]
    y = df['Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    lr_r2 = float(r2_score(y_test, y_pred_lr))
    lr_metrics = {
        'r2': lr_r2,
        'mae': float(mean_absolute_error(y_test, y_pred_lr)),
        'mse': float(mean_squared_error(y_test, y_pred_lr))
    }

    # Train Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_r2 = float(r2_score(y_test, y_pred_rf))
    rf_metrics = {
        'r2': rf_r2,
        'mae': float(mean_absolute_error(y_test, y_pred_rf)),
        'mse': float(mean_squared_error(y_test, y_pred_rf))
    }

    # Select best model based on R² score
    if rf_r2 > lr_r2:
        best_model = rf_model
        best_model_name = "Random Forest"
        best_metrics = rf_metrics
    else:
        best_model = lr_model
        best_model_name = "Linear Regression"
        best_metrics = lr_metrics

    return {
        'model': best_model,
        'model_name': best_model_name,
        'le_gender': le_gender,
        'le_degree': le_degree,
        'le_job': le_job,
        'scaler_age': scaler_age,
        'scaler_exp': scaler_exp,
        'metrics': best_metrics,
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'df': df
    }


def make_prediction(artifacts: dict, age: float, gender: str, degree: str, job: str, exp_years: float) -> float:
    # transform numeric
    age_scaled = artifacts['scaler_age'].transform([[age]])[0][0]
    exp_scaled = artifacts['scaler_exp'].transform([[exp_years]])[0][0]

    # transform categoricals; if unseen label, raise a clear error
    try:
        gender_enc = int(artifacts['le_gender'].transform([gender])[0])
        degree_enc = int(artifacts['le_degree'].transform([degree])[0])
        job_enc = int(artifacts['le_job'].transform([job])[0])
    except Exception as e:
        raise ValueError(f"Categorical value not seen during training: {e}")

    X_in = np.array([[age_scaled, gender_enc, degree_enc, job_enc, exp_scaled]])
    pred = artifacts['model'].predict(X_in)[0]
    return float(pred)


def main():
    st.title("Employee Salary Prediction")
    st.write("App trains both Linear Regression and Random Forest models, and deploys the one with better precision (R² Score).")

    st.sidebar.header("Data / Training")
    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
    use_default = not uploaded

    data_path = "Dataset09-Employee-salary-prediction.csv"
    if uploaded:
        st.sidebar.success("Using uploaded file")
        df = pd.read_csv(uploaded)
        # try to normalize columns if necessary
        if list(df.columns) != ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_years', 'Salary']:
            try:
                df.columns = ['Age', 'Gender', 'Degree', 'Job_Title', 'Experience_years', 'Salary']
            except Exception:
                st.sidebar.error("Uploaded CSV doesn't match expected format. Expected columns: Age, Gender, Degree, Job_Title, Experience_years, Salary")
                st.stop()
    else:
        try:
            df = load_data(data_path)
        except FileNotFoundError:
            st.error(f"Default dataset not found at {data_path}. Upload a CSV or place the dataset in the app folder.")
            st.stop()

    st.sidebar.write(f"Rows available for training: {df.shape[0]}")

    # Train model
    with st.spinner("Training model..."):
        artifacts = train_model(df)

    st.sidebar.success("Model trained")
    
    # Display model comparison
    st.subheader("Model Performance Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Linear Regression**")
        lr_m = artifacts['lr_metrics']
        st.write(f"R² Score: {lr_m['r2']:.4f}")
        st.write(f"MAE: {lr_m['mae']:.2f}")
        st.write(f"MSE: {lr_m['mse']:.2f}")
    
    with col2:
        st.write("**Random Forest**")
        rf_m = artifacts['rf_metrics']
        st.write(f"R² Score: {rf_m['r2']:.4f}")
        st.write(f"MAE: {rf_m['mae']:.2f}")
        st.write(f"MSE: {rf_m['mse']:.2f}")
    
    st.markdown("---")
    st.info(f"🚀 **Selected Model for Deployment: {artifacts['model_name']}** (Better R² Score)")
    
    st.subheader("Selected Model Metrics (Test Set)")
    m = artifacts['metrics']
    st.write(f"R² Score: {m['r2']:.4f}")
    st.write(f"MAE: {m['mae']:.2f}")
    st.write(f"MSE: {m['mse']:.2f}")

    st.subheader("Make a prediction")
    col1, col2 = st.columns(2)
    unique_genders = list(artifacts['df']['Gender'].unique())
    unique_degrees = list(artifacts['df']['Degree'].unique())
    unique_jobs = list(artifacts['df']['Job_Title'].unique())

    with col1:
        age = st.number_input("Age", min_value=16, max_value=100, value=int(artifacts['df']['Age'].median()))
        gender = st.selectbox("Gender", unique_genders)
        degree = st.selectbox("Degree", unique_degrees)

    with col2:
        job = st.selectbox("Job Title", unique_jobs)
        exp_years = st.number_input("Experience (years)", min_value=0.0, max_value=80.0, value=float(artifacts['df']['Experience_years'].median()))

    if st.button("Predict Salary"):
        try:
            pred = make_prediction(artifacts, float(age), gender, degree, job, float(exp_years))
            st.success(f"Predicted salary: {pred:,.2f}")
        except ValueError as err:
            st.error(str(err))

    st.markdown("---")
    st.subheader("Dataset preview")
    st.dataframe(artifacts['df'].head(20))


if __name__ == '__main__':
    main()
