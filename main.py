from fastapi import FastAPI, Request, Form, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import numpy as np
import pandas as pd
import pickle

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def index(request:Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict', response_class=HTMLResponse)
def create_id(request: Request,pregnancies: int = Form(...), glucose: int = Form(...), bloodpressure: int = Form(...), skinthickness: int = Form(...), insulin: int = Form(...), bmi: str = Form(...), diabetiespedegreefunction: int = Form(...), age: int = Form(...)):
    Pregnancies = pregnancies
    Glucose = glucose
    BloodPressure = bloodpressure
    SkinThickness = skinthickness
    Insulin = insulin
    BMI = bmi
    DiabetesPedigreeFunction = diabetiespedegreefunction
    Age = age

    filename = 'modelForPrediction.sav'
    loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
    scalar = pickle.load(open("sandardScalar.sav", 'rb'))
    # predictions using the loaded model file
    prediction = loaded_model.predict(scalar.transform(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]))
    if prediction ==[1]:
            prediction = "diabetes"

    else:
            prediction = "Normal"

    # showing the prediction results in a UI
    if  prediction =="diabetes":

         return templates.TemplateResponse('diabetes.html', {'request': request, 'prediction':prediction})
    else:
         return templates.TemplateResponse('Normal.html', {'request': request, 'prediction':prediction})


