import pandas as pd 
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
import numpy as np 
from textblob import TextBlob

def create_dummies(dataset, column):
    dummies = pd.get_dummies(dataset[column], prefix = column)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset


def is_junk(string):
    string = string.split()
    not_letter = 0
    hashtags = 0
    nothing = 0
    total_length = 1
    for word in string:
        total_length += len(word)
        for char in word:
            if not char.isalpha():
                not_letter += 1
            if char == "#":
                hashtags +=1
            if not char.isalpha() and not char.isdigit():
                nothing = nothing + 1

    if len(string)/total_length > 15:
        return True
    elif hashtags > 3:
        return True
    elif nothing / total_length > 0.2:
        return True
    elif not_letter / total_length > 0.4:
        return True
    else:
        return False

def validate_column(dataset, column):
    dataset[column] = dataset[column].fillna("None")
    for i in range(dataset.shape[0]):
        dataset[column][i] = is_junk(dataset[column][i])
    return dataset