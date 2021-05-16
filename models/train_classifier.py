import sys
# import libraries
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report, hamming_loss, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

SEED = 42
TEST_SIZE = 0.2

def load_data(database_filepath):
    '''
    DESCRIPTION:
    A function to load the data for training the model
        
    INPUT:
    database_filepath - the database table to input the data

    OUTPUT:
    X - a dataframe with the messages data to be trained
    Y - a dataframe with the target columns
    category_names - a list with the target columns names
    '''    
    
    url = 'sqlite:///' + str(database_filepath)
    df = pd.read_sql_table('messages', url)  
    print(df.columns)
    X = df['message']
    Y = df.drop(columns = ['message', 'original', 'genre', 'related']) 
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    DESCRIPTION:
    A function to transform the text data from the disaster messages
        
    INPUT:
    text - the disaster message
    
    OUTPUT:
    tokens - cleaned and lemmatized parts of the disaster message
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    DESCRIPTION:
    A function to build the classifier model pipeline
           
    OUTPUT:
    cv - the model pipeline with the parameters set for gridsearch
    '''
    
    pipeline = Pipeline([
            ('features', FeatureUnion([           
                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ]))
            ])),
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=SEED, n_jobs=-1), n_jobs=-1))
        ])
    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 200],
        'clf__estimator__min_samples_split': [4, 8],
        'clf__estimator__max_depth': [5,8]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    DESCRIPTION:
    A function to evaluate the trained model with the test dataset
        
    INPUT:
    model - the trained model
    X_test - the features from test dataset
    Y_test - the targets from test dataset
    category_names - a list with the target columns names
    
    OUTPUT:
    Metrics for the test dataset
    '''
    
    Y_pred = model.predict(X_test)
    print('Hamming Loss:', hamming_loss(Y_test, Y_pred))
    print('Accuracy:', accuracy_score(Y_test, Y_pred))

    
def save_model(model, model_filepath):
    '''
    DESCRIPTION:
    A function to persist the trained classifier model 
    
    INPUT:
    model - the trained model
    model_filepath - path where the model is saved
           
    OUTPUT:
    pipeline and trained model saved in pickle format
    '''
    
    model_fp = open(model_filepath, 'wb')
    #return pickle.dumps(model, model_filepath)
    return pickle.dump(model, model_fp)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=SEED, test_size=TEST_SIZE, shuffle=True)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()