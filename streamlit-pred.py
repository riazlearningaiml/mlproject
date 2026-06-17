import streamlit as st
import pandas as pd
import requests
import subprocess
import time

# Starting FAST API server
def start_fastapi():
    try:
        requests.get("http://127.0.0.1:8000", timeout=2)
    except requests.exceptions.ConnectionError:
        subprocess.Popen(["uvicorn", "app:app", "--reload"])
        time.sleep(3)

start_fastapi()
st.set_page_config(page_title="Student Performance Prediction")
st.title("Student Performance Input Form")

# Gender
gender = st.selectbox(
    "Gender",
    ["female", "male"]
)

# Race / Ethnicity
race_ethnicity = st.selectbox(
    "Race / Ethnicity",
    [
        "group A",
        "group B",
        "group C",
        "group D",
        "group E"
    ]
)

# Parent Education
parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school"
    ]
)

# Lunch
lunch = st.selectbox(
    "Lunch",
    ["standard", "free/reduced"]
)

# Test Preparation Course
test_preparation_course = st.selectbox(
    "Test Preparation Course",
    ["none", "completed"]
)

# Scores
math_score = st.number_input(
    "Math Score",
    min_value=0,
    max_value=100,
    value=91
)

reading_score = st.number_input(
    "Reading Score",
    min_value=0,
    max_value=100,
    value=86
)

# Submit Button
if st.button("Submit"):

    input_data = {
       "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course,
        "math_score": math_score,
        "reading_score": reading_score
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=input_data)

    if response.status_code == 200:
        prediction = response.json()["prediction"]
        st.success(f"Predicted Writing Score: {prediction:.2f}")
    else:
        st.error("Prediction failed")

    