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
    df = df.sample(frac=.1)

    # let's build X and y
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])

    return X, y, y.keys()


def tokenize(text):

    nltk.download(['punkt', 'stopwords', 'wordnet']);

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
    ngram_range = (1, 2)

    # TfidfTransformer
    norm = 'l1'
    smooth_idf = True
    sublinear_tf = False
    use_idf = True

    # RandomForestClassifier
    criterion = 'entropy'
    min_samples_leaf = 2
    n_estimators = 100

    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(max_df=max_df, max_features=max_features, min_df=min_df, ngram_range=ngram_range, tokenizer=tokenize)),

            ('tfidf', TfidfTransformer(norm=norm, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, use_idf=use_idf)),

            ('clf', MultiOutputClassifier(RandomForestClassifier(criterion=criterion, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators)))
        ]
    )

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)

    y_pred_df = pd.DataFrame(data=y_pred, columns=category_names)

    for name in Y_test:
        print(name)
        print(classification_report(Y_test[name], y_pred_df[name]))


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

        category_names = Y_test.keys()

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
