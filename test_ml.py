from sklearn.externals import joblib
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
from sklearn.svm import SVC, LinearSVC
#from features.SparseInteractions import SparseInteractions

from newml import to_vector

model = gensim.models.Word2Vec.load("receipt_word2vec")

texts = ["body spray", "ciment", "Tooth Brush"]
vectors = np.array([to_vector(i, model) for i in texts])
pl = joblib.load('ml_per_receipt_nn.pkl')
print("result from neutral network")
print(pl.predict(vectors))

pl = joblib.load('ml_per_receipt_random_forest.pkl')
print("result from random forest")
print(pl.predict(vectors))

pl = joblib.load('ml_per_receipt_linear_svc.pkl')
print("result from linear svc")
print(pl.predict(vectors))
