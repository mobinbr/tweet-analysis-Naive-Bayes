# Naive Bayes 3-class Classifier 

class NaiveBayesClassifier:

    def __init__(self, classes):
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  
        self.classes = classes
        self.class_word_counts = None
        self.class_counts = None
        self.vocab = None

    def train(self, data):
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            pass

    def calculate_prior(self):
        # calculate log prior
        return None 

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        # return the corresponding value
        return None

    def classify(self, features):
        # predict the class
        # inputs: features(list) --> words of a tweet 
        best_class = None
        return best_class
    