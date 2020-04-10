import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load the Disaster Messages Data and the Categories Date
    
    PARAMETERS:
        messages_filepath (str): Path to the CSV file containing messages
        categories_filepath (str): Path to the CSV file containing categories
    RETURN:
        df (dataframe like): Combined data containing messages and categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    return df 


def clean_data(df):
    """
    Clean Categories Data Function
    
    PARAMETERS:
        df (DataFrame): Combined data containing messages and categories
    RETURN:
        df (DataFrame): Combined data containing messages and categories with categories cleaned up
    """
    
    # Spliting the categories to respective columns
    categories = df['categories'].str.split(';',expand=True)
    
    #Fixing the categories columns name
    row = categories.iloc[0]
    category_colnames = [category_name.split('-')[0] for category_name in row]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(np.int)
    
    df = df.drop('categories',axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    """
    Saves Processed Data to SQLite Database using the sqlalchemy library
    
    PARAMETER:
        df (DataFrame): Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)  # creating the database engine

    # creating a table name using the database_filename
    table_name = database_filename.replace(".db","") + "_table"

    # converting dataframe to sql and saving to the database
    df.to_sql(table_name, engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()