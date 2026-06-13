import streamlit as st
import pandas as pd
from src.utils import load_object

st.set_page_config(page_title="Student Performance Prediction")

st.title("Student Performance Input Form")

model  = load_object('artifacts/model.pkl')
preprocess  = load_object('artifacts/preprocessor.pkl')

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

    input_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race_ethnicity],
        "parental level of education": [parental_level_of_education],
        "lunch": [lunch],
        "test preparation course": [test_preparation_course],
        "math score": [math_score],
        "reading score": [reading_score]
    })

    X_transformed = preprocess.transform(input_data)

    prediction = model.predict(X_transformed)

    st.write("Prediction:", prediction[0])
    
    