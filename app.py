from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-data', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'), 
            ethnicity = request.form.get('ethnicity'), 
            parent_education = request.form.get('parent_education'), 
            lunch = request.form.get('lunch'), 
            test_course = request.form.get('test_course'), 
            writing_score = request.form.get('writing_score'), 
            reading_score = request.form.get('reading_score')
        )

        df = data.get_data_ass_dataframe()
        print(df)

        predict_pipeline = PredictPipeline()
        predicted_data = predict_pipeline.predict(df)
        return render_template('form.html', results=predicted_data[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)