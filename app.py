import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

@st.cache(allow_output_mutation=True)
def load_model():
    df = pd.read_csv("empsalpred.csv")
    df = df.dropna()

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("salary", axis=1)
    y = df["salary"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le, X.columns.tolist()

model, le, features = load_model()

st.set_page_config(page_title="Salary Prediction", layout="centered")
st.title("💼 Employee Salary Prediction")

with st.form("form"):
    age = st.slider("Age", 18, 70, 30)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "State-gov"])
    education = st.selectbox("Education", ["Bachelors", "Masters", "PhD"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Exec-managerial", "Prof-specialty", "Adm-clerical", "Sales"])
    hours = st.slider("Hours per week", 1, 100, 40)
    native_country = st.selectbox("Native Country", ["United-States"])
    submit = st.form_submit_button("Predict")

if submit:
    input_dict = {
        "age": age,
        "workclass": le.fit_transform([workclass])[0],
        "education": le.fit_transform([education])[0],
        "occupation": le.fit_transform([occupation])[0],
        "hours-per-week": hours,
        "native-country": le.fit_transform([native_country])[0]
    }
    input_df = pd.DataFrame([input_dict])[features]

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")

    input_df["Predicted Salary (₹)"] = prediction
    csv = input_df.to_csv(index=False)
    st.download_button("📥 Download Result", data=csv, file_name="predicted_salary.csv", mime="text/csv")
