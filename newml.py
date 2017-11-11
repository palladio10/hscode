import gensim
import pandas as pd
import re
import sys
import platform

import numpy as np
#import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sparse_interactions import SparseInteractions






chi_k = 300
#TOKEN = '[A-Za-z]+(?=\\s+)'
TOKEN = '[A-Za-z0-9]+'
TOKEN_RE = re.compile('[A-Za-z0-9]+')

ID = ['SdcReceiptSignature']
LABELS = ['hs_code']
NUMERIC_COLUMNS = ['total']
TEXT_COLUMNS = ['text']
CATEGORICAL_COLUMNS = ['tax_rate']
FEATURES = TEXT_COLUMNS

SIMILAR_COUNT = 80

def add_weight(weight_dict, key, weight):
    if key in weight_dict:
        weight_dict[key] = max(weight_dict[key], weight)
    else:
        weight_dict[key] = weight

def get_weight(weight_dict, key):
    if key in weight_dict:
        return weight_dict[key]
    else:
        return 0

def to_vector(text, model, similar_count=SIMILAR_COUNT):
    names = TOKEN_RE.findall(text)
    weight_dict = dict()
    for name in names:
        if name in model.wv:
            add_weight(weight_dict, name, 1)
            similarities = model.most_similar(name, topn=similar_count)
            for t in similarities:
                add_weight(weight_dict, t[0], t[1])
    result = []
    for i in range(len(model.wv.vocab)):
        result.append(get_weight(weight_dict, model.wv.index2word[i]))
    return np.array(result)

def toSerial(dataframe, model, similar_count=SIMILAR_COUNT, to_keep=TEXT_COLUMNS):
    to_drop = list(set(dataframe.columns) - set(to_keep))
    text_data = dataframe.drop(to_drop, axis=1)
    text_data.fillna('', inplace=True)
    result = []
    for index, row in text_data.iterrows():
        result.append(to_vector(row['text'], model, similar_count))
    return np.array(result)

def prepare_df():
    df = pd.read_excel("./trainingdata/sample_data_tags.xlsx")
    df.set_index(ID, inplace=True)
    df.dropna(subset=[LABELS], inplace=True)
    df = df[LABELS + FEATURES]
    df = df.loc[df['hs_code'] > 0]
    df = df.drop_duplicates(subset=LABELS+FEATURES)
    #df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].apply(lambda x: x.astype('category'))

    df = filter_df(df)

    df.to_csv("tts.csv");

def filter_df(df):
    ## Drop if only one example of the hs_code (set up to handle HS and UNSPSC together)
    df['count'] = 1
    groups = df[LABELS +['count']].groupby(LABELS).count()
    groups = groups[(groups['count'] == 1)].reset_index()
    idx = df[LABELS].apply(lambda x: ~x.isin(list(groups[x.name]))).product(axis=1).astype('bool')
    df = df[idx]

    groups = df[LABELS +['count']].groupby(LABELS).count()
    del df['count']
    return df

def run_ml(similar_count=SIMILAR_COUNT):
    #import logging
    #logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    model = gensim.models.Word2Vec.load("receipt_word2vec")

    df = pd.read_csv("tts.csv", delimiter=',')
    df = filter_df(df)

    df_train, df_test = train_test_split(df,
                                         train_size=0.8,
                                         random_state=321,
                                         stratify=df[LABELS])
    X_train, y_train = toSerial(df_train[FEATURES], model, similar_count), df_train[LABELS]
    X_test, y_test = toSerial(df_test[FEATURES], model, similar_count), df_test[LABELS]

    pl = Pipeline([
        #('selector', get_text_data),
        #('vectorizer', CountVectorizer(token_pattern=TOKEN,
        #                               binary=True,
        #                               ngram_range=(1, 3))),
        #('dim_red', SelectKBest(chi2, chi_k)),

        #('int', SparseInteractions(degree=2)),  ## This is very cpu costly but adds 20ppt on accuracy
        #('clf', LinearSVC())
        ('clf', LogisticRegression()) ## all ofthis is running single cpu # 89% when 420 similarity
        #('clf', SVC())
        #('clf', RandomForestClassifier(n_estimators=15))
        #('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1))

    ])
    # fit the pipeline to our training data
    pl.fit(X_train, np.ravel(y_train.values))

    #from sklearn.externals import joblib
    #joblib.dump(pl, 'ml_per_receipt_nn.pkl')

    # Compute and print accuracy:
    accuracy_train = pl.score(X_train, y_train)
    accuracy_test = pl.score(X_test, y_test)
    print('\nAccuracy on training dataset: ', accuracy_train,
          '\nAccuracy on test dataset: ', accuracy_test)

    '''X_all, y_all = toSerial(df[FEATURES], model, similar_count), df[LABELS]
    pl.fit(X_all, np.ravel(y_all.values))
    accuracy_all = pl.score(X_all, y_all)
    print('\nAccuracy on all dataset: ', accuracy_all)'''


if __name__ == '__main__':
    #prepare_df()
    for i in range(1000, 2000):
        print("------i={}------".format(i))
        run_ml(i * 10)
