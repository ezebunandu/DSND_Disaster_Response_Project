# Disaster Response Pipeline
This project was completed as part of Udacity's Data Science Nanodegree Program

Classifier App for disaster response messages. Includes the following modules

* ETL pipeline to load and clean the training data and save to an sqlite db (etl_pipeline.py)
* ML pipeline that trains a classifier model on the cleaned trainiing data in the db (ml_pipeline.py)
* Flask wrapper for classifying new messages using the trained model (app/run.py)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/etl_pipeline.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/