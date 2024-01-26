# Naive Bayes 3-class Classifier 
import math

class NaiveBayesClassifier:

    def __init__(self, classes):
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  
        self.classes = classes
        self.class_word_counts = {word: {'positive': 0, 'negative': 0, 'neutral': 0} for word in set()}
        # lambda_pos -> pos/neg, lambda_neg -> neg/neut, lambda_neut -> neut/pos
        self.lambda_table = {word: {'positive': 0, 'negative': 0, 'neutral': 0, 'lambda_pos': 0, 'lambda_neg': 0, 'lambda_neut': 0} for word in set()}
        self.class_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        self.vocab = set()

    def train(self, data):
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            for word in features:
                self.class_counts[label] += 1
                self.vocab.add(word)
                for c in self.classes:
                    self.class_word_counts[word][c] += 1

        for word in self.vocab:
            for c in self.classes:
                self.lambda_table[word][c] = self.calculate_likelihood(word, c)

            self.lambda_table[word]['lambda_pos'] = math.log(self.lambda_table[word]['positive']/self.lambda_table[word]['negative'])
            self.lambda_table[word]['lambda_neg'] = math.log(self.lambda_table[word]['negative']/self.lambda_table[word]['neutral'])
            self.lambda_table[word]['lambda_neut'] = math.log(self.lambda_table[word]['neutral']/self.lambda_table[word]['positive'])

    def calculate_prior(self):
        # calculate log prior
        # you can add some attributes to this method
  
        # Your Code
        return None 

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        # return the corresponding value

        return (self.class_word_counts[word][label] + 1) / (self.class_counts[label] + len(self.vocab))

    def classify(self, features):
        # predict the class
        # inputs: features(list) --> words of a tweet 
        best_class = None 

        # Your Code
        return best_class
    