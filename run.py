from template import NaiveBayesClassifier


def preprocess(tweet_string):
    # clean the data and tokenize it
    features = []
    return features

def load_data(data_path):
    # load the csv file and return the data
    data = []
    return data


# train your model and report the duration time
train_data_path = "train_data.csv"
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))

test_string = "I love playing football"

print(nb_classifier.classify(preprocess(test_string)))
