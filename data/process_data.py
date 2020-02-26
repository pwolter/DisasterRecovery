"""process_data.py performs the ETL pipeline for this project. It does:

- Loads the `messages.csv` and `categories.csv` files and transforms
  them into pandas dataframes
- Merges the two dataframes
- Cleans the data
- Stores it in a SQLite database
"""

__version__ = '0.1'
__author__ = 'Pablo Wolter'


import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Reads 'disaster_messages.csv' and 'disaster_categories.csv' into
    pnadas dataframes. Merges them into a single dataframe and returns it.

    Args:
    messages_filepath:
        Full path for the disaster_messages csv file.
    categories_filepath:
        Full path for the disaster_categories csv file.
    Returns:
    df:
        Merged Dataframe.
    """


    # load messages dataset
    messages = pd.read_csv('disaster_messages.csv')

    # load categories dataset
    categories = pd.read_csv('disaster_categories.csv')

    # merge datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """Reads the dataframe, splits the categories individualy, renames
    columns, converts the categorical data to integers and returns it.

    Args:
    df:
        Dataframe to clean.
    Returns:
    df:
        Cleaned Dataframe.
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0].reset_index(drop=True)

    # use this row to extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to integers
    for column in categories:
        # set each value to be the last character of the string an
        # dconvert it to int
        categories[column] = categories[column].astype(str) \
                             .apply(lambda x : int(x[-1:]))

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new
    # `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # replace related values = 2 to 0 according to documentation
    df['related'] = df['related'].replace(2, 0)

    # delete duplicates
    return df[~df.duplicated()]


def save_data(df, database_filename):
    """Reads the dataframe, and saves it in a SQLite database.

    Args:
    df:
        Dataframe to clean.
    Returns:
    None:
        Saves Dataframe to a SQLite database.
    """

    # database and table name
    database_name = 'sqlite:///{}'.format(database_filename)
    table_name = database_filename.replace('.db', '')

    engine = create_engine(database_name)
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
