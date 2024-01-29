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
        self.class_word_counts = {}
        self.class_word_likelihood = {}
        self.prior = {}
        self.general_count = 0
        self.class_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        self.vocab = set()

    def train(self, data):
        # training process:
        # inputs: data(list) --> each item of list is a tuple 
        # the first index of the tuple is a list of words and the second index is the label(positive, negative, or neutral)

        for features, label in data:
            self.class_counts[label] += 1
            for word in features:
                self.vocab.add(word)
                if word not in self.class_word_counts:
                    self.class_word_counts[word] = {'positive': 0, 'negative': 0, 'neutral': 0}
                self.class_word_counts[word][label] += 1

        for word in self.vocab:
            self.class_word_likelihood[word] = {'positive': 0, 'negative': 0, 'neutral': 0}
            for c in self.classes:
                self.class_word_likelihood[word][c] = self.calculate_likelihood(word, c)
                
        self.general_count = sum(self.class_counts.values())
        self.calculate_prior()

    def calculate_prior(self):
        # calculate log prior
        for c in self.classes:
            self.prior[c] = math.log(self.class_counts[c]/self.general_count)

    def calculate_likelihood(self, word, label):
        # calculate likelihhood: P(word | label)
        word_count = self.class_word_counts[word][label] if word in self.vocab else 0
        return math.log((word_count + 1) / (self.class_counts[label] + len(self.vocab)))

    def classify(self, features):
        # predict the class
        best_class = None
        class_vals = {}
        for c in self.classes:
            class_vals[c] = 0
            for word in features:
                if word in self.vocab:
                    class_vals[c] += self.class_word_likelihood[word][c]
                else:
                    class_vals[c] += self.calculate_likelihood(word, c)
            class_vals[c] += self.prior[c]
        
        best_class = max(class_vals, key=class_vals.get)
        return best_class
    