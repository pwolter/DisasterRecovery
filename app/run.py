"""run.py Runs the Flask application
"""

__version__ = '0.1'
__author__ = 'Pablo Wolter'


import os
import re
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

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



# load data
database_filepath = '../data/DisasterResponse.db'

db_engine_path = 'sqlite:///{}'.format(database_filepath)

# get database table name
table_name = os.path.split(database_filepath)[1].replace('.db', '')

engine = create_engine(db_engine_path)
df = pd.read_sql_table(table_name, engine)

# load model
model_path = "../models/classifier.pkl"
model = joblib.load(model_path)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # english - non-english messages percentage
    values = []
    values.append(df['original'].count())
    values.append(df['original'].isnull().sum())
    labels = ['English', 'No English']

    # frequency of words histogram
    df_test = df.drop(columns=['id', 'message', 'original'])
    genre_keys = df_test.aggregate(sum).iloc[1:].reset_index() \
        .sort_values(by=0, ascending=False)['index']
    genre_values = df_test.aggregate(sum).iloc[1:].reset_index() \
        .sort_values(by=0, ascending=False)[0]


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=genre_keys,
                    y=genre_values
                )
            ],

            'layout': {
                'title': 'Distribution of Messages',
                'yaxis': {
                    'title': "Count"
                },
            }
        },

        {
            'data': [
                Pie(
                    labels=labels,
                    values=values
                )
            ],

            'layout': {
                'title': 'Distribution of Message (English - No english)',
            }
        },

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
