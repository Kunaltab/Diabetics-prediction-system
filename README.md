# Diabetes Prediction App

A simple Streamlit app that predicts diabetes risk from medical input values using a machine learning model trained on the Pima Indians Diabetes dataset.

## What is included

- `app.py` - Streamlit web app
- `diabetes_model.py` - model training script
- `diabetes.csv` - dataset used for training
- `diabetes_model.pkl` - saved trained model

## Features

- Predicts whether a person is likely to have diabetes
- Shows an estimated probability when available
- Uses a Random Forest classifier
- Easy to run locally with Streamlit

## Requirements

- Python 3.10+
- `pip`
- Packages listed in `requirements.txt`

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

If you want to retrain the model, run:

```bash
python diabetes_model.py
```

This will recreate `diabetes_model.pkl`.

## Run the app

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Input fields

The app uses these inputs:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

## Project structure

```text
diabetes-predictor/
├── app.py
├── diabetes_model.py
├── diabetes.csv
├── diabetes_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Notes

- This project is for educational purposes only.
- It is not a medical diagnosis tool.
- If `diabetes_model.pkl` is missing, run `python diabetes_model.py` first.
