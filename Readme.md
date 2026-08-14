# Student Exam Performance Prediction (End-to-End ML Project)

This project predicts a student's math score using demographic and academic context features:

- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course
- reading_score
- writing_score

The project includes:

- notebook-based EDA and model experimentation
- a modular training pipeline
- model and preprocessor artifact saving
- a Flask web app for real-time prediction

## 1. Project Overview

The workflow follows a standard machine learning lifecycle:

1. Data ingestion from source CSV
2. Train-test split
3. Data preprocessing with scikit-learn pipelines
4. Model selection and hyperparameter tuning
5. Model artifact persistence
6. Inference through a Flask app UI

Primary dataset source used in notebook:

- Kaggle: Students Performance in Exams

Current source file used by ingestion:

- notebook/data/stud.csv

## 2. Tech Stack

- Python
- pandas, numpy
- scikit-learn
- catboost, xgboost
- dill
- Flask (+ gunicorn in requirements)

## 3. Repository Structure

```
.
|-- app.py
|-- Readme.md
|-- requirements.txt
|-- setup.py
|-- artifacts/
|   |-- data.csv
|   |-- train.csv
|   |-- test.csv
|   |-- model.pkl
|   `-- preprocessor.pkl
|-- src/
|   |-- components/
|   |   |-- data_ingestion.py
|   |   |-- data_tranformation.py
|   |   `-- model_trainer.py
|   |-- pipeline/
|   |   |-- predict_pipeline.py
|   |   `-- train_pipeline.py
|   |-- exception.py
|   |-- logger.py
|   `-- utils.py
|-- templates/
|   |-- index.html
|   `-- form.html
|-- static/
|   `-- style.css
|-- notebook/
|   |-- EDA Student Performance.ipynb
|   |-- Model Training.ipynb
|   `-- data/stud.csv
|-- catboost_info/
`-- logs/
```

## 4. Installation

### 4.1 Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4.2 Install dependencies

```powershell
pip install -r requirements.txt
```

Optional editable install:

```powershell
pip install -e .
```

## 5. Model Training Pipeline

Training entry point currently runs from:

- src/components/data_ingestion.py

When executed directly, it performs:

1. Read raw data from notebook/data/stud.csv
2. Save raw copy to artifacts/data.csv
3. Split into train/test and save as artifacts/train.csv and artifacts/test.csv
4. Transform features using ColumnTransformer in data_tranformation.py
5. Train and evaluate multiple regressors in model_trainer.py
6. Save:
	- artifacts/preprocessor.pkl
	- artifacts/model.pkl

Run training:

```powershell
python src/components/data_ingestion.py
```

## 6. Models Considered

The trainer evaluates these regressors:

- LinearRegression
- KNeighborsRegressor
- DecisionTreeRegressor
- RandomForestRegressor
- AdaBoostRegressor
- GradientBoostingRegressor
- XGBRegressor
- CatBoostRegressor

Model selection criterion:

- best test R2 score from GridSearchCV-tuned candidates

## 7. Web App (Inference)

Flask app entry point:

- app.py

Routes:

- / : landing page
- /predict-data : prediction form (GET/POST)

The prediction flow:

1. User submits form fields
2. CustomData converts form values to a pandas DataFrame
3. PredictPipeline loads artifacts/preprocessor.pkl and artifacts/model.pkl
4. Preprocess + predict
5. Predicted math score rendered on form page

Run locally:

```powershell
python app.py
```

Then open:

- http://127.0.0.1:5000

## 8. Notebooks

- notebook/EDA Student Performance.ipynb: problem understanding, checks, EDA, plots
- notebook/Model Training.ipynb: feature prep and baseline model experimentation

## 9. Generated Outputs

After training/inference runs, you should expect:

- artifacts/*.csv for split datasets
- artifacts/model.pkl and artifacts/preprocessor.pkl
- logs/*.log for runtime logging
- catboost_info/* from CatBoost training runs

## 10. Notes

- File names currently use the spelling data_tranformation.py (project-internal usage is consistent with this name).
- src/pipeline/train_pipeline.py is present but currently empty.

## 11. Next Improvements

- add a dedicated training orchestrator in src/pipeline/train_pipeline.py
- add unit tests for preprocessing and prediction pipeline
- add model evaluation report export (metrics + chosen hyperparameters)
- improve deployment readiness (production WSGI config, Dockerfile, CI)
