from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# A combination of Word lemmatization + LinearSVC model

trainfile = pd.read_json("/Users/nikunjpatel/Desktop/Data Mining/DataMining Grad Project/train.json")

trainfile['ingredients_clean_string'] = [' , '.join(z).strip() for z in trainfile['ingredients']]
trainfile['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in trainfile['ingredients']]       

testfile = pd.read_json("/Users/nikunjpatel/Desktop/Data Mining/DataMining Grad Project/test.json") 

testfile['ingredients_clean_string'] = [' , '.join(z).strip() for z in testfile['ingredients']]
testfile['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in testfile['ingredients']]       

corpustr = trainfile['ingredients_string']
vectorizertr = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word", 
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)
tfidftr = vectorizertr.fit_transform(corpustr).todense()
corpusts = testfile['ingredients_string']
vectorizerts = TfidfVectorizer(stop_words='english')
tfidfts = vectorizertr.transform(corpusts)

predictor_train = tfidftr

target_data = trainfile['cuisine']

predictor_test = tfidfts

trainfile.to_csv("train_cleanFile.csv", index=False) 
testfile.to_csv("test_cleanFile.csv", index=False) 

#KNN theorem
neigh = KNeighborsClassifier(n_neighbors=3,algorithm='brute')
classifier=neigh.fit(predictor_train,target_data)
predictions=classifier.predict(predictor_test)
testfile['cuisine'] = predictions
testfile = testfile.sort('id' , ascending=True)
testfile[['id' ,'ingredients_string' ,'cuisine' ]].to_csv("knn-Test_pred.csv", index=False) 

#Naive-Bayes Theorem
nb = MultinomialNB()
nbClassifier = nb.fit(predictor_train,target_data);
predictions = nbClassifier.predict(predictor_test)
testfile['cuisine'] = predictions
testfile = testfile.sort('id' , ascending=True)
testfile[['id' ,'ingredients_string', 'cuisine' ]].to_csv("naive-test_pred.csv", index=False)


#logistic Regression
lr = LogisticRegression()
lrClassifier = lr.fit(predictor_train,target_data)
predictions = lrClassifier.predict(predictor_test)
testfile['cuisine'] = predictions
testfile = testfile.sort('id' , ascending=True)
testfile[['id' ,'ingredients_string', 'cuisine' ]].to_csv("Logistic-test_pred.csv", index=False)
