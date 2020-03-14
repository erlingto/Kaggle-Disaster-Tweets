import pandas as pd 
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
import numpy as np 
from textblob import TextBlob
import utilities
import pickle 

def train_LogisticRegression():
    model = LogisticRegression()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    train = utilities.validate_column(train, "location")
    test = utilities.validate_column(test, "location")

    train = utilities.create_dummies(train, "keyword")
    test = utilities.create_dummies(test, "keyword")
    

    testsentiments = [None] * test.shape[0]
    sentiments = [None] * train.shape[0]

    for j in range(train.shape[0]):
        text = train["text"][j]
        blob = TextBlob(text)

        sentiments[j] = blob.sentiment.polarity
       
    
    for t in range(test.shape[0]):
        testtext = test["text"][t]
        testblob = TextBlob(testtext)

        testsentiments[t] = testblob.sentiment.polarity

    train["sentiment"] = sentiments
    test["sentiment"] = testsentiments
    
    train = utilities.validate_column(train, "text")
    test = utilities.validate_column(test, "text")

    x = train.drop(['target', 'keyword'], axis= 1)
    y = train['target']

    model.fit(x, y)

    test = test.drop(['keyword'], axis= 1)
    print(test.columns)
    print(train.columns)

    test.to_csv("test_data", index = False)

    predictions = model.predict(test)

    filename = 'trained_LogRes.sav'
    pickle.dump(model, open(filename, 'wb'))
    submission = pd.DataFrame({"id": test.id, "target": predictions})
    submission.to_csv("LogRes_Submissions.csv", index = False)

train_LogisticRegression()
