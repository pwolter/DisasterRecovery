"""train_classifier.py performs the ML pipeline for this project. It does:

- Loads data from the SQLite database saved before
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file
"""

__version__ = '0.1'
__author__ = 'Pablo Wolter'


import sys
import os
from sqlalchemy import create_engine
import pandas as pd

import re
import nltk
import argparse
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

from sklearn.utils import parallel_backend


def load_data(database_filepath):
    """Reads 'disaster_messages.csv' and 'disaster_categories.csv' into
    pandas dataframes. Merges them into a single dataframe and returns it.

    Args:
    messages_filepath:
        Full path for the disaster_messages csv file.
    categories_filepath:
        Full path for the disaster_categories csv file.
    Returns:
    df:
        Merged Dataframe.
    """

    # ../data/DisasterResponse.db
    db_engine_path = 'sqlite:///{}'.format(database_filepath)
    # get database table name
    table_name = os.path.split(database_filepath)[1].replace('.db', '')


    # load data from database
    engine = create_engine(db_engine_path)
    conn = engine.connect()
    df = pd.read_sql_table(table_name, conn)

    # let's sample the data
    df = df.sample(frac = sample_percentage, random_state = random_state)

    # columns to drop from dataframe
    # plus 2 columns child_alone and shops that are all 0
    columns=['id', 'message', 'original', 'genre',
             'request', 'offer', 'child_alone', 'shops']

    # let's build X and y
    X = df['message']
    y = df.drop(columns = columns)
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):
    """Replaces URLs by placeholders, removes punctuation, lowercase all words,
    tokenize and lemmetize the tokens.

    Args:
    text:
        Pandas series with the text to be tokenized/transformed.
    Returns:
    lemmed:
        Text ready to be input into ML algorithms to build a model.
    """

    # URL replacement
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Lowercase and punctuation removal
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()

    # Stop word removal
    stop_words = set(stopwords.words("english"))
    words = [w for w in text if w not in stop_words]

    # Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmed


def build_model():
    """Build a model based on Scikit-learn's GridSearchCV.

    Args:
    None:
        Builds the model.
    Returns:
    best_model:
        best model found by GridsearchCV.
    """

    parameters = {
        'vect__ngram_range': ((1, 2), (2, 3)),

        'tfidf__use_idf': ('True', 'False'),

        'clf__estimator__n_estimators': (10, 30, 50),
        'clf__estimator__warm_start': ('True',),
        'clf__estimator__min_samples_split': (0.6, 0.7, 0.8),
    }

    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(
                RandomForestClassifier(random_state = random_state))
            )
        ]
    )

    best_model = GridSearchCV(estimator = pipeline, param_grid = parameters,
                              n_jobs = -1, cv = cv, refit = True,
                              return_train_score = True, verbose = 1)

    return best_model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the model's performance using Scikit-learn's
    classification_report.

    Args:
    model:
        Model to be evaluated.
    X_test:
        Unseen data for the model to make predictions.
    Y_test:
        Ground truth to measure the model's performance.
    category_names:
        String with the category names to report performance on.
    Returns:
    None:
        Prints out performance metrics.
    """

    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):
    """Saves the best model into a pickle file that will be used by the
    Flask application to make predictions.

    Args:
    model:
        best model found by GridserachGV.
    Returns:
    model_filepath:
        Full path for the pickle file to save.
    """

    file_name = model_filepath

    with open(file_name, 'wb') as pickled_model:
        pickle.dump(model, pickled_model)


def arguments_parser():
    """Parse the arguments for the script.

    Args:
    None
    Returns:
    parse_args() object with parsed arguments dict
    """

    parser = argparse.ArgumentParser(
        description='''Please provide the filepath of the disaster messages
        database as the first argument and the filepath of the pickle file to
        save the model to as the second argument.''',
        epilog='''Example: python train_classifier.py
        ../data/DisasterResponse.db classifier.pkl'''
    )

    parser.add_argument('database', help='Database file path')
    parser.add_argument('model', help='Filename to save the pickle model')

    return parser.parse_args()


def main():
    args = arguments_parser()

    model_filepath = args.model
    database_filepath = args.database

    # moved thsi line here so it loads only when arparse succeds
    nltk.download(['punkt', 'stopwords', 'wordnet'])

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))

    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    # have to add parallel_backend because I was having an error
    # with pickle
    with parallel_backend('multiprocessing'):
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Model Parameters...')
        print(model.best_params_)
        #model = model.best_estimator_

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    # Sample percentage of dataframe to speed up modeling
    # changed to 100% for full run
    sample_percentage = 1

    # setup cross validation parameter globaly
    cv = 10

    # setup random_state globaly as well
    random_state = 42

    main()
