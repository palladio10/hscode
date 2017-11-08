import gensim
import pandas as pd
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
from sklearn.svm import SVC, LinearSVC
#from features.SparseInteractions import SparseInteractions
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

model = gensim.models.Word2Vec.load("receipt_word2vec")

df = pd.read_excel("./trainingdata/sample_data_tags.xlsx")

ID = ['SdcReceiptSignature']
LABELS = ['hs_code']
NUMERIC_COLUMNS = ['total']
TEXT_COLUMNS = ['text']
CATEGORICAL_COLUMNS = ['tax_rate']
FEATURES = TEXT_COLUMNS

df.set_index(ID, inplace=True)
df.dropna(subset=[LABELS], inplace=True)
df = df[LABELS + FEATURES]
#df[CATEGORICAL_COLUMNS] = df[CATEGORICAL_COLUMNS].apply(lambda x: x.astype('category'))

## Drop if only one example of the hs_code (set up to handle HS and UNSPSC together)
df['count'] = 1
groups = df[LABELS +['count']].groupby(LABELS).count()
groups = groups[(groups['count'] == 1)].reset_index()
idx = df[LABELS].apply(lambda x: ~x.isin(list(groups[x.name]))).product(axis=1).astype('bool')
df = df[idx]

groups = df[LABELS +['count']].groupby(LABELS).count()
del df['count']

df.to_csv("test.csv");

df_train, df_test = train_test_split(df,
                                     train_size=0.8,
                                     random_state=123,
                                     stratify=df[LABELS])

X_train, y_train = df_train[FEATURES], df_train[LABELS]
X_test, y_test = df_test[FEATURES], df_test[LABELS]

print(X_train)

chi_k = 300
#TOKEN = '[A-Za-z]+(?=\\s+)'
TOKEN = '[A-Za-z0-9]+'

def combine_text_columns(dataframe, to_keep=TEXT_COLUMNS):
    ''' Combines all of the text columns into a single vector that has all of
        the text for a row.'''
    print(dataframe)
    to_drop = list(set(dataframe.columns) - set(to_keep))
    text_data = dataframe.drop(to_drop, axis=1)
    text_data.fillna('', inplace=True)
    return text_data.apply(lambda x: ' '.join(x), axis=1)

get_text_data = FunctionTransformer(combine_text_columns, validate=False)

pl = Pipeline([
    ('selector', get_text_data),
    ('vectorizer', CountVectorizer(token_pattern=TOKEN,
                                               binary=True,
                                               ngram_range=(1, 3))),
    #('dim_red', SelectKBest(chi2, chi_k)),

    #('int', SparseInteractions(degree=2)),  ## This is very cpu costly but adds 20ppt on accuracy
    #        ('clf', LinearSVC())
    ('clf', LogisticRegression()) ## all ofthis is running single cpu
    #        ('clf', SVC())
           #('clf', RandomForestClassifier(n_estimators=15))
])

# fit the pipeline to our training data
pl.fit(X_train, np.ravel(y_train.values))

# Compute and print accuracy:
accuracy_train = pl.score(X_train, y_train)
accuracy_test = pl.score(X_test, y_test)
print('\nAccuracy on training dataset: ', accuracy_train,
      '\nAccuracy on test dataset: ', accuracy_test)
