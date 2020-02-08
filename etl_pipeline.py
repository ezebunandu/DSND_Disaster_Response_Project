import sqlite3
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads the messages and categories csv files

    arg:
        messages_filepath: path to the messages csv
        categories_filepth: path to the categories csv

    returns:
            concatenated dataframe of messages and categories
            joined on the id column
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how="inner", on='id')
    return df


def clean_data(df):
    """cleans the dataframe to include one-hot categories
    and remove duplicates

    args:
        df: dataframe to clean
        categories: the dataframe containing the full category information
    """
    categories = df['categories'].str.split(';', expand=True)

    # use first row as column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])

    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True, ignore_index=True)

    return df


def save_data(df, database_filename, table="messages"):
    """
    saves a dataframe to a database table

    args:
        df: the dataframe to save
        database_filename: the database to save to
        table to save to: default "messages"

    returns:
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    conn = sqlite3.connect('{}'.format(database_filename))

    cur = conn.cursor()
    cur.execute(f'DROP TABLE IF EXISTS {table}')
    conn.commit()
    conn.close()

    df.to_sql(table, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )

        df = load_data(messages_filepath, categories_filepath)
        print("Cleaning data...")

        df = clean_data(df)
        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)
        print("Cleaned data saved to database!")

    else:
        print(
            """
        Please provide the filepaths of the messages and categories
        datasets as the first and second argument respectively, as
        well as the filepath of the database to save the cleaned data
        to as the third argument. \n\nExample: python process_data.py 
        disaster_messages.csv disaster_categories.csv
        DisasterResponse.db"""
        )


if __name__ == "__main__":
    main()
