Customer Churn Prediction Project
Project Overview

This project focuses on predicting customer churn using machine learning techniques. The aim is to identify customers who are likely to discontinue a service, helping businesses take proactive retention actions.

The project follows a modular, pipeline-based architecture and includes a Flask web application for prediction.

Project Structure
project/
│
├── artifacts/                     # Intermediate files & trained models
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl                  # Trained ML model
│   └── preprocessor.pkl           # Data preprocessing pipeline
│
├── notebook/                      # Jupyter notebooks
│   └── data/
│       └── stud.csv               # Raw dataset
│
├── src/                           # Source code
│   ├── components/                # Core ML pipeline scripts
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/                  # Training & prediction pipelines
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py               # Custom exceptions
│   ├── logger.py                  # Logging utility
│   └── utils.py                   # Helper functions
│
├── templates/                     # HTML templates (Flask)
│
├── app.py                         # Flask application entry point
├── requirements.txt               # Project dependencies
├── setup.py                       # Package setup
└── README.md

Flow Chart (Project Pipeline)
flowchart TD
    A[Start] --> B[Data Ingestion]
    B --> C[Data Transformation]
    C --> D[Model Training]
    D --> E[Model Evaluation]
    E --> F[Save Model & Artifacts]
    F --> G[Prediction Pipeline]
    G --> H[Churn Prediction Output]

Workflow Explanation
1. Data Ingestion

Loads the raw dataset

Splits data into training and testing sets

Saves processed files into the artifacts/ directory

2. Data Transformation

Handles missing values

Encodes categorical features

Scales numerical features

Saves preprocessing pipeline as preprocessor.pkl

3. Model Training

Trains a machine learning model

Evaluates model performance

Saves trained model as model.pkl

4. Prediction Pipeline

Loads saved model and preprocessor

Generates churn predictions for new inputs

Integrated with Flask application

Installation & Setup
1. Clone the repository
git clone https://github.com/Keerthivardhan1507/mlproject.git
cd mlproject

2. Create & activate Conda environment
conda create -n myenv python=3.8 -y
conda activate myenv

3. Install dependencies
pip install -r requirements.txt

Train the Model
python src/pipeline/train_pipeline.py

Run Predictions (Pipeline)
python src/pipeline/predict_pipeline.py

Run Flask Application
python app.py


Then open:

http://127.0.0.1:5000/

Technologies Used

Python 3.8

Pandas, NumPy

Scikit-learn

CatBoost

Flask

Modular ML Pipeline Architecture

Future Enhancements

Docker deployment

CI/CD integration

Model monitoring & drift detection

Cloud deployment (AWS / Azure)