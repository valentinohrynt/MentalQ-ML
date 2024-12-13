# MentalQ - C242-PS246

## Table of Contents

1. [Description](#Description)
2. [Team](#C242-PS246---ml)
3. [Technology & Library](#Technology-&-Library)
4. [Requirement](#Requirement)
5. [Installation Steps](#Installation-Steps)
6. [Usage of Flask Application](#Usage-of-Flask-Application)

## Description
The **Machine Learning** part of the MentalQ app helps analyze users' mental health by studying their daily notes. It uses a **LSTM model** to look at the text and understand how users are feeling. The process starts by collecting and cleaning the data, then training the model to recognize different mental health patterns. Once the model is ready, it’s built into the app to give users personalized feedback based on their mood. Tools like **TensorFlow** and **NLTK** help build and improve the model. By using this technology, MentalQ can help users better understand their mental health, offering support when they need it most.

## C242-PS246 - ML

| Bangkit ID | Name | Learning Path | University | LinkedIn |
| ---      | ---       | ---       | ---       | ---       |
| M129B4KX2462 | Melinda Naurah Salsabila | Machine Learning | Politeknik Negeri Jember | [![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/melinda-naurah/) |
| M227B4KY3579 | Rafi Achmad Fahreza | Machine Learning | Universitas Jember | [![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafiachmadfr/) |
| M129B4KY1504 | Fikri Faqih Al Fawwaz | Machine Learning | Politeknik Negeri Jember | [![text](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/fikrifaqihalfawwaz/) |

## Technology & Library

- Pandas
- Matplotlib
- NumPy
- Random
- Stanza
- re (Regular Expressions)
- H5py
- Pickle
- JSON
- TensorFlow
- Keras
- Scikit-learn

## Requirement
Make sure you have installed:

Python 3.9 or newer
pip (package installer for Python)

## Installation Steps

1. Clone the Repository
Clone the repository from GitHub to your computer.

```bash
git clone https://github.com/MentalQ-App/MentalQ-Model.git
cd model_save_ml/ml_model_lstm.h5
```

2. Ensure Model and Data Availability
Make sure the model file (model.h5) is available in the project's root directory.

## Usage of Flask Application

This application uses Flask as a web framework to create prediction API using a LSTM model. Several dependencies must be installed before running the application.

Usage of Flask Application
API Endpoint
/predict
Method: POST

Request Format:
Send a JSON payload with a list of statements.

```bash
    "statements": [ "Saya merasa hidup saya tidak berarti dan lebih baik mati saja.",
    ]
```

Respon Format :
The API returns a list with the predicted mental health status and confidence scores for each statement.

```bash
        "confidence_scores": {
            "Anxiety": 0.007962707430124283,
            "Bipolar": 0.004331799224019051,
            "Depression": 0.33362269401550293,
            "Normal": 0.003008028957992792,
            "Personality disorder": 0.008331895805895329,
            "Stress": 0.005370507948100567,
            "Suicidal": 0.6373724341392517
        },
        "predicted_status": "Suicidal",
        "statement": "Saya merasa hidup saya tidak berarti dan lebih baik mati saja."
```

