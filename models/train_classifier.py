import sys
import os
from sqlalchemy import create_engine
import pandas as pd

import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

from sklearn.utils import parallel_backend

nltk.download(['punkt', 'stopwords', 'wordnet'])

def load_data(database_filepath):

    # ../data/DisasterResponse.db
    db_engine_path = 'sqlite:///{}'.format(database_filepath)
    # get database table name
    table_name = os.path.split(database_filepath)[1].replace('.db', '')


    # load data from database
    engine = create_engine(db_engine_path)
    conn = engine.connect()
    df = pd.read_sql_table(table_name, conn)

    # Sample 25% of dataframe to speed up modeling - change to 100% for full run
    sample_percentage = 0.025
    df = df.sample(frac=sample_percentage)

    # let's build X and y
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre', 'request', 'offer'])
    category_names = y.columns.tolist()

    return X, y, category_names


def tokenize(text):

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

    # CountVectorizer
    max_df = 0.75
    max_features = 5000
    min_df = 1
    ngram_range = (2, 3)

    # TfidfTransformer
    norm = 'l1'
    smooth_idf = True
    sublinear_tf = False
    use_idf = True

    # RandomForestClassifier
    # criterion = 'entropy'
    # min_samples_leaf = 2
    # n_estimators = 100

    parameters = {
        # 'vect__max_df': (0.75, 0.8),
        # 'vect__max_features': (5000, 10000, 50000),
        # 'vect__ngram_range': ((1, 2), (2, 2)),

        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),

        'clf__estimator__n_estimators': (100, 150, 200),
        # 'clf__estimator__criterion': ('gini', 'entropy'),
        'clf__estimator__min_samples_leaf': (2, 4, 6),
    }

    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(max_df=max_df, max_features=max_features, min_df=min_df, ngram_range=ngram_range, tokenizer=tokenize)),

            ('tfidf', TfidfTransformer(norm=norm, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, use_idf=use_idf)),

            # ('clf', MultiOutputClassifier(RandomForestClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)))

            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )

    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1, cv=5, refit=True, return_train_score=True, verbose=1)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)

    print(classification_report(Y_test, y_pred, target_names = category_names))


def save_model(model, model_filepath):

    file_name = model_filepath

    with open(file_name, 'wb') as pickled_model:
        pickle.dump(model, pickled_model)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

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

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
