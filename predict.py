# api/predict.py
import joblib
import os
import json

model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'xgb_model3.pkl')
model = joblib.load(model_path)

def handler(request):
    try:
        body = request.json()
        features = [
            body['age'],
            body['avg_glucose_level'],
            body['hypertension'],
            body['heart_disease'],
            body['bmi']
        ]
        prediction = model.predict([features])
        return {
            "statusCode": 200,
            "body": json.dumps({"prediction": int(prediction[0])})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
