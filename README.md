# Tweet Sentiment Analysis

## Overview

This project implements a Naive Bayes classifier for sentiment analysis on tweets. The classifier is trained on a dataset containing labeled tweets categorized as positive, negative, or neutral sentiments. After training, the classifier can analyze new tweets and classify them into one of these sentiment categories.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Data Files](#data-files)
- [Algorithm Details](#algorithm-details)
- [Customization](#customization)

## Introduction

The project consists of two main files: `run.py` and `template.py`. The `run.py` script contains the main functionality to train, test, evaluate accuracy, and analyze tweets using the Naive Bayes classifier implemented in `template.py`.

## Installation

 To run the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/mobinbr/tweet-sentiment-analysis
   ```

2. Navigate to the project directory:
    ```bash
    cd tweet-sentiment-analysis
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your data: <br>
Ensure you have the following CSV files:<br>
`train_data.csv`: Contains labeled training data.<br>
`eval_data.csv`: Contains evaluation data for calculating accuracy.<br>
`test_data_nolabel.csv`: Contains unlabeled test data to be classified.

2. Run the training script:

    ```python
    python run.py
    ```

This will train the Naive Bayes classifier on the provided training data, evaluate its accuracy using the evaluation data, and generate a classification result for the test data.

3. Analyze tweets: <br>
After running the script, you can analyze new tweets by calling the classify method of the trained classifier. For example:

    ```python
    test_string = "I love playing football"
    print(nb_classifier.classify(preprocess(test_string)))
    ```

## Files

- `run.py`: Script for preprocessing data, training the classifier, evaluating accuracy, and analyzing tweets.
- `template.py`: Implementation of the NaiveBayesClassifier class for sentiment analysis.

## Data Files
`train_data.csv`: Labeled training data containing tweets and sentiment labels.
`eval_data.csv`: Labeled evaluation data for calculating accuracy.
`test_data_nolabel.csv`: Unlabeled test data for testing the classifier.

## Algorithm Details

The sentiment analysis algorithm is based on the Naive Bayes classifier. Key components include:

**Preprocessing**: Tweets are preprocessed to remove noise, such as special characters and stopwords, and tokenized into words for analysis.

**Training**: The Naive Bayes classifier is trained using labeled tweet data. It calculates the likelihood of each word occurring in each sentiment category and determines the prior probabilities of each category.

**Classification**: Given a new tweet, the classifier calculates the probability of it belonging to each sentiment category based on the trained model and assigns it to the category with the highest probability.

## Customization

You can customize the project by adjusting parameters in the `run.py` file, such as file paths for data, and by modifying hyperparameters in the `template.py` file, such as smoothing techniques or feature selection methods in the Naive Bayes classifier. Experiment with different datasets and preprocessing techniques to improve the accuracy of sentiment analysis.