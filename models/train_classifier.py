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

    # ../data/DisasterResponse.db
    db_engine_path = 'sqlite:///{}'.format(database_filepath)
    # get database table name
    table_name = os.path.split(database_filepath)[1].replace('.db', '')


    # load data from database
    engine = create_engine(db_engine_path)
    conn = engine.connect()
    df = pd.read_sql_table(table_name, conn)

    # Sample 25% of dataframe to speed up modeling - change to 100% for full run
    sample_percentage = 0.1
    df = df.sample(frac=sample_percentage)

    # let's build X and y
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre', 'request', 'offer', 'related'])
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
    words = [w for w in text if w not in stopwords.words("english")]

    # Stemming
    stemmed = [PorterStemmer().stem(w) for w in words]

    # Lemmatization
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]

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

    # CountVectorizer
    # max_df = 0.8
    # max_features = 5000
    # min_df = 1
    # ngram_range = (3, 3)

    # TfidfTransformer
    # norm = 'l1'
    # smooth_idf = True
    # sublinear_tf = False
    # use_idf = True

    # RandomForestClassifier
    # criterion = 'entropy'
    # min_samples_leaf = 2
    # n_estimators = 100

    parameters = {
        'vect__max_df': (0.75, 0.8, 0.85),
        'vect__max_features': (5000, 10000, 50000),
        'vect__ngram_range': ((1, 2), (2, 3)),

        # 'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),

        'clf__estimator__n_estimators': (50, 100, 150),
        # 'clf__estimator__criterion': ('gini', 'entropy'),
        'clf__estimator__learning_rate': (0.2, 0.3, 0.4),
    }

    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            # ('vect', CountVectorizer(max_df=max_df, max_features=max_features,
            # min_df=min_df, ngram_range=ngram_range, tokenizer=tokenize)),

            ('tfidf', TfidfTransformer(use_idf=True)),
            # ('tfidf', TfidfTransformer(norm=norm, smooth_idf=smooth_idf,
            # sublinear_tf=sublinear_tf, use_idf=use_idf)),

            # ('clf', MultiOutputClassifier(RandomForestClassifier(
            # criterion=criterion, min_samples_leaf=min_samples_leaf,
            # n_estimators=n_estimators)))
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ]
    )

    best_model = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1, cv=5, refit=True, return_train_score=True, verbose=1)

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    with parallel_backend('multiprocessing'):
        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
