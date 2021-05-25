# Disaster Response Pipeline Project

### Motivation:
When hundreds, thousands messages are sent after a disaster, there's a need to treat them properly so they can be directed to the organizations that can help in a specific aspect. This project aims to deliver a web app that classify messages into categories in order to speed the process of identifying a request and offer help after a disaster.


### Data and File Description:

#### 1. Data:
The data used in this projet was provided by Figure Eight as a part of Udacity Nanodegree Project.  
Folder "data":  
    - messages.csv: contains the data about the message that was sent (id, message, genre) 
    - categories.csv: contains the data about the categories the message belonged to (request, offer, medical assistance, food and others)  
The file "process data.py" loads the two datasets, clean the data and creates a database.
    
#### 2. Model:
Folder "model':  
    - train_classifier.py: process the text message and train a classifier model
    
#### 3. Web App:
Folder "app":  
    - run.py: runs a flask application to classify new messages using the trained model. It also display some visuals from the train dataset.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgments

[Building a Disaster Response Web-Application](https://towardsdatascience.com/building-a-disaster-response-web-application-4066e6f90072)

This project was developed as part of the requirements for Data Scientist Udacity Nanodegree. 
