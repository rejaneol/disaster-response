import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath) 
    categories = pd.read_csv(categories_filepath) 
    # merge datasets
    df = messages.merge(categories, on = 'id').set_index('id')
    df.drop_duplicates(inplace=True)
    #print('loaded:', df.shape, df.columns)
    #print(df.head())
    return df


def clean_data(df):
    # transform categories dataset
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # create the categories columns names
    row = categories.iloc[0]
    category_colnames = row.str.split('-').str.get(0)
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # merge datasets
    df = df.join(categories).drop(columns = 'categories')

    # drop duplicates
    df.drop_duplicates(inplace=True)
    #print('cleaned:', df.shape)
    #print(df.head())
    
    return df


def save_data(df, database_filename):
    url = 'sqlite:///' + str(database_filename)
    engine = create_engine(url)
    df.to_sql('messages', engine, index=False, if_exists='replace')  #if_exists='append'


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