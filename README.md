# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Identification of credit card customers that are most likely to churn.

## Files and data description
Overview of the files and data present in the root directory. 
```
.
├── Guide.ipynb          # Getting started and troubleshooting tips
├── churn_notebook.ipynb # Contains the code to be refactored
├── churn_library.py     # Defines the functions
├── churn_script_logging_and_tests.py # Done: Finish tests and logs
├── README.md            # Provides project overview, and instructions to use the code
├── data                 # Save data in a `data` folder
│   └── bank_data.csv
├── images               # Store EDA & training results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models
```
The data file bank_data.csv can be downloaded on [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

## Running Files
In order to run the files, make sure you install the correct libraries with the
following steps:
1. cd to the directory where requirements.txt is located.
2. activate your virtualenv, with a version of Python superior or equal to 3.8
3. run: `python -m pip install -r requirements.txt` in your shell.

Then, to train the model, run `python churn_library.py`; this script will:
1. perform the EDA, and save the results in `images/eda`
2. perform feature engineering by encoding categorical variable and selecting features
3. train the model, and save the models in `models/`
4. make predictions
5. evaluate the model based on the predictions, and save the results in `images/results`

In order to test and log the results of each function, follow these steps:
1. make sure the constants in `constants.py` are correct
2. run `python churn_script_logging_and_tests.py`
3. Open `logs/churn_library.log` to view the result from running all tests.