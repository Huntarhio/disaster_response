# import libraries
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, average_precision_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, SVC
import pickle

def load_data(database_filepath):
    '''
    PARAMETERS: 
        database_filepath (str) : Filepath used for importing the database     
    RETURNS
        Returns the following variables:
        X(Pandas DataFrame) : Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y (Pandas DataFrame) : Returns the categories of the dataset.  This will be used for classification based off of the input X
        y.keys (list) : Just returning the columns of the Y columns
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    #converting the class in related to just 0 and 1
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)

    #dropping the child_alone column since all values are zero and holds no information
    df.drop('child_alone', axis=1, inplace=True)
    X = df.message.values
    y = df.iloc[:,5:]
    return X, y, y.keys()

def tokenize(text):
    '''
    PARAMETERS 
        text (str): Text to be processed   
    RETURN
        Returns a processed text variable that was tokenized, lower cased, stripped, and lemmatized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        """
        This methos checks if first word in sentence is a verb or not

        PARAMETERS 
            text (str): Text to be processed   
        RETURN
            Returns (bool): a boolean to confirm if first word in sentence is a verb or not
        """

        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))  #obtaing the pos taggings

            #selecting the pos tag of the first item in the sentence
            first_word, first_tag = pos_tags[0] 
            
            #checking if the first word in the senntece has a pos tagging of VERB
            if first_tag in ['VB', 'VBP'] or first_word == 'RT': 
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        This method alpplies the starting_verb method a series contained with texts

        PARAMETERS 
            X (series): Text to be processed   
        RETURN
            Returns (DataFrame): A data frame with boolean as its values
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model(X_train,y_train):
    '''
    INPUT 
        X_Train: Training features for use by GridSearchCV
        y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    '''
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    
    c_s = np.linspace(0.1,1.2,12)

    parameters = {
        'clf__estimator__C': c_s
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, 
                            n_jobs=-1, verbose=3, scoring='accuracy')
    cv.fit(X_train,y_train)
    return cv

def evaluate_model(pipeline, X_test, Y_test, category_names):
    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        y_test: Label features, testing set
        category_names: List of the categories 
    OUTPUT
        This method does nto specifically return any data to its calling method.
        However, it prints out the precision, recall and f1-score
    '''
    # predict on test data
    y_pred = pipeline.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))
    print(accuracy_score(Y_test, y_pred))

def save_model(model, model_filepath):
    '''
    Saves the model to disk
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        While there is no specific item that is returned to its calling method, this method will save the model as a pickle file.
    '''    
    temp_pickle = open(model_filepath, 'wb')
    pickle.dump(model, temp_pickle)
    temp_pickle.close()

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=45)
        
        print('Building model...')
        model = build_model(X_train,Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print(model.best_params_)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        
        ###WILL NEED TO CXLEAN THIS UP
        print('TYPE OF MODEL')
        print(type(model))
        
        
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