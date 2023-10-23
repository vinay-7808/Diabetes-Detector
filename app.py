from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained logistic regression model
svc_model = joblib.load('svc_model.pkl')

class CustomerData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict_DiabeticOrNot/")
async def predict_DiabeticOrNot(data: CustomerData):
    input_data = data.dict()
    input_features = np.array(list(input_data.values())).reshape(1, -1)
    prediction = svc_model.predict(input_features)
    if(prediction[0] > 0.5):
      prediction = 'Diabetic'
    else:
      prediction = 'Non_Diabetic'
    return {
       "DiabetesPredicton": prediction
      }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)