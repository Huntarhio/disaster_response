import os
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import nltk
import re

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

stop_words = set(stopwords.words('english'))
from collections import Counter

app = Flask(__name__)
DATABASE_URL = os.environ.get('DATABASE_URL')


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def count_word(data):
    text = ''.join(data)
    new_text = re.sub('[^a-zA-Z]+', ' ', text)
    word_tokens = word_tokenize(''.join(new_text))

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

    counts = (Counter(filtered_sentence))
    counts = dict(counts)

    words = counts.keys()
    count = counts.values()
    counts_df = pd.DataFrame(zip(words, count), columns=['words', 'counts'])

    counts_gr_500 = counts_df[counts_df['counts'] > 500]
    counts_list = list(counts_gr_500['counts'].values)
    words_list = list(counts_gr_500['words'].values)

    return counts_list, words_list


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)

# load models
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    messages = df['message']

    counts = count_word(messages)

    genre_names = list(genre_counts.index)

    category_names = df.iloc[:, 4:].columns
    category_boolean = (df.iloc[:, 4:] != 0).sum().values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # GRAPH 1 - genre graph
        {
            'data': [
                Bar(
                    x=counts[1],
                    y=counts[0],
                    marker_color='''dimgray, dimgrey, dodgerblue, firebrick,
                        floralwhite, forestgreen, fuchsia, gainsboro,
                        ghostwhite, gold, goldenrod, gray, grey, green,
                        greenyellow, honeydew, hotpink, indianred, indigo,
                        ivory, lavender, lavenderblush, lawngreen,
                        lemonchiffon, lightblue, lightcoral, lightcyan,
                        lightgoldenrodyellow, lightgray, lightgrey,
                        lightgreen, lightpink, lightsalmon, lightseagreen,
                        lightskyblue, lightslategray
                    '''.replace(',', '').split()
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'tickangle': 35
                }
            }
        },

        # GRAPH 2 - word count graph
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker_color=['maroon', 'lightpink', 'fuchsia']
                )
            ],

            'layout': {
                'title': 'Distribution of Most Common words',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        },

        # GRAPH 3 - category graph
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_boolean,
                    marker_color='''dimgray, dimgrey, dodgerblue, firebrick,
                        floralwhite, forestgreen, fuchsia, gainsboro,
                        ghostwhite, gold, goldenrod, gray, grey, green,
                        greenyellow, honeydew, hotpink, indianred, indigo,
                        ivory, lavender, lavenderblush, lawngreen,
                        lemonchiffon, lightblue, lightcoral, lightcyan,
                        lightgoldenrodyellow, lightgray, lightgrey,
                        lightgreen, lightpink, lightsalmon, lightseagreen,
                        lightskyblue, lightslategray
                    '''.replace(',', '').split()
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        }
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

engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse_table', engine)
