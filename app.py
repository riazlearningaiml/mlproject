import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

class StudentData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    math_score: int
    reading_score: int



@app.get("/")
def home():
    return {"message": "Welcome to the ML Prediction API"}

@app.post("/predict")
def predict(student_data: StudentData):
    data = pd.DataFrame([{
        "gender": student_data.gender,
        "race/ethnicity": student_data.race_ethnicity,
        "parental level of education": student_data.parental_level_of_education,
        "lunch": student_data.lunch,
        "test preparation course": student_data.test_preparation_course,
        "math score": student_data.math_score,
        "reading score": student_data.reading_score
    }])
    X=preprocessor.transform(data)
    prediction=model.predict(X)

    return {"prediction": int(prediction[0])}


    
