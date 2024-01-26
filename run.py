from template import NaiveBayesClassifier
import csv
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
from textblob import TextBlob
import numpy as np

def preprocess(tweet_string):
    # clean the data and tokenize it
    features = []
    tweet_string = str(tweet_string)

    # normalizing
    tweet_string = tweet_string.lower()

    # removing unicode characters
    tweet_string = re.sub(r"(@[^\s]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|http.+", "", tweet_string)

    # spell checking
    tweet_string = TextBlob(tweet_string)
    checked_string = str(tweet_string.correct())

    # removing stopwords
    stop_words = set(stopwords.words('english'))
    toks = word_tokenize(checked_string)
    # np.concatenate(stop_words, ['?'], [''])
    filtered_toks = [word for word in toks if word not in stop_words]


    # stemming & lemmatization
    # lemmatizer = WordNetLemmatizer()
    # tweet_string = [lemmatizer.lemmatize(word) for word in filtered_toks]

    features = set(filtered_toks)
    return features

def load_data(data_path):
    # load the csv file and return the data
    data = []
    with open(data_path, mode ='r') as file:
        csvFile = csv.reader(file)
        next(csvFile)
        for lines in csvFile:
            tweet = preprocess(str(lines[2]))
            data.append((tweet, int(lines[3])))
    return data


# train your model and report the duration time
train_data_path = "train_data.csv"
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

test_string = "I love playing football"

print(nb_classifier.classify(preprocess(test_string)))
