import kaggle
import kagglehub
import numpy as np
import pandas as pd
from IPython.display import display
from kagglehub import KaggleDatasetAdapter
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

def LR_OVA_prob(xtest,threshold,classifier):
    y_pred_prob = classifier.predict_proba(xtest)
    t=threshold
    y_pred_new = (y_pred_prob >= t).astype(int)
    return y_pred_new

def print_report(y_true, y_pred,avg_type,metric, mlb):
    report=classification_report(y_true, y_pred, target_names=mlb.classes_,output_dict=True)
    metric_val=report[avg_type][metric]
    print(avg_type,metric,"=",metric_val)
    return metric_val

def result_table(y_true,y_pred,df,mlb):
    actual_genres = mlb.inverse_transform(y_true)
    predicted_genres = mlb.inverse_transform(y_pred)
    # Create a new DataFrame to display results
    results_df = pd.DataFrame({
    'Synopsis': df['synopsis'],
    'Actual Genres': actual_genres,
    'Predicted Genres': predicted_genres
    })
    return results_df

def hit_rate(y_true, y_pred):
    hits = np.logical_and(y_true, y_pred).sum(axis=1) > 0
    return hits.mean()


# __all__ = [
#     'kaggle', 
#     'kagglehub', 
#     'np', 
#     'pd', 
#     'display', 
#     'KaggleDatasetAdapter', 
#     'Counter', 
#     'plt', 
#     'MultiLabelBinarizer',
#     'CountVectorizer',
#     'iterative_train_test_split',
#     'OneVsRestClassifier',
#     'LogisticRegression',
#     'classification_report',
#     'jaccard_score',
#     'TfidfVectorizer'

# ]