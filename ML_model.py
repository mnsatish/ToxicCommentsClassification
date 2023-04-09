"""
Machine Learning Logistic Regression Multi-label Classification Model
"""

import numpy as np
import pandas as pd
import random
import torch
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
import pickle

DEFAULT_SEED: int = 9898
random.seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)
torch.cuda.manual_seed(DEFAULT_SEED)

# initialize TFIDF
tfidf_vec = TfidfVectorizer(min_df=5, max_features=200000)
# initialize classifier chains multi-label classifier
model = ClassifierChain(LogisticRegression(solver='lbfgs', max_iter=400))

def classification_model_train(comments_df):
    categories = list(comments_df.columns.values)
    categories = categories[2:]

    toxic_comments = comments_df[comments_df[categories].sum(axis=1) > 0]
    clean_comments = comments_df[comments_df[categories].sum(axis=1) == 0]

    reduced_comments_df = pd.concat([toxic_comments, clean_comments.sample(25_000)])

    print("Reduced data size: ", reduced_comments_df.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(reduced_comments_df.comment_text.values.astype('U'),
                                                        reduced_comments_df[categories], test_size=0.3, random_state=20)

    tfidf = tfidf_vec.fit_transform(X_train)
    X_tfidf = pd.DataFrame(tfidf.todense())
    X_tfidf.head(2)
    
    # Training logistic regression model on train data
    model.fit(X_tfidf, Y_train)
    
    # Model evaluation on validation set
    test_data = tfidf_vec.transform(X_test)
    test_data = pd.DataFrame(test_data.todense())
    print(test_data.head())
    
    # predict
    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data)
    print("ROC AUC = ", metrics.roc_auc_score(Y_test, probabilities, average="macro"))
    print("Accuracy = ", metrics.accuracy_score(Y_test, predictions))

    # save the model
    filename = 'final_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # inp = ["fuck you man!"]
    # test_phrase = tfidf_vec.fit_transform(inp)
    # test_phrase = pd.DataFrame(test_phrase.todense())
    # model.predict(test_phrase)
    

def classification_model_test(comments_df):
    X_test = tfidf_vec.transform(comments_df.comment_text.values.astype('U'))
    X_test = pd.DataFrame(X_test.todense())
    Y_test = comments_df

    # predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    print("ROC AUC = ", metrics.roc_auc_score(Y_test, probabilities, average="macro"))
    print("Accuracy = ", metrics.accuracy_score(Y_test, predictions))
    
