# Disaster Response Pipeline Project (Udacity - Data Science Nanodegree)


## Table of Contents
1. [Description](#description)
2. [Getting Started](#getting_started)
	1. [Dependencies](#dependencies)
	2. [Installing](#installation)
	3. [Executing Program](#execution)
	4. [Files](#files)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)
6. [Reference](#reference)


## Description<a name="descripton"></a>

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to use data engineering, Natural Language Processing (NLP) and MachineLearning Pipeline to categorize messages on a real time basis.

This project is divided in the following key sections:

1. Using ETL pipepline to extract data from source, transform the data and load them into a SQLite DB using sqlachemy
2. Build a machine learning pipeline to train a model which is then used to classify messages in 36 categories
3. Run a web app which can show model results in real time


## Getting Started<a name="getting_started"></a>


### Dependencies<a name="dependencies"></a>
* Python 3.5+
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Model Loading and Saving Library: Pickle
* Web App and Data Visualization: Flask, Plotly


### Installing<a name="installation"></a>
To clone the git repository:
```
git clone https://github.com/Huntarhio/disaster_response.git
```

### Executing Program<a name="execution"></a>:
1. You can run the following commands in the project's directory to set up the database, train model and save the model.

    - To run ETL pipeline to clean data and store the processed data in the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db`
    - To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file
        `python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. In the web interface, enter a disaster message. For Example ("There is a need for medication here")

5. Click on classify message button and wait for your message to be classified

6. Scroll down to view classification. The classification are the the item highlighted in green



### Files<a name="files"></a>
**app/templates/**: templates/html files for web app

**data/process_data.py**: Extract Train Load (ETL) pipeline used for reading data from source, data cleaning/transformation, feature extraction/engineering, and loading data in a SQLite database

**models/train_classifier.py**: A machine learning pipeline that loads a cleaned data, trains a model, and saves the trained model as a .pkl file for later use

**models/classifier.pkl** the saved model after training

**run.py**: This file can be used to launch the Flask web app used to classify disaster messages


## Authors<a name="authors"></a>

* [Adeoti Adegboyega](https://github.com/huntarhio)

## Reference<a name="reference"></a>

* [Naveen Setia](https://github.com/canaveensetia)


## License<a name="license"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Acknowledgements<a name="acknowledgement"></a>

* [Udacity](https://www.udacity.com/) for providing an amazing Data Science Nanodegree Program
* [Figure Eight](https://www.figure-eight.com/) for providing the relevant dataset to train the model

