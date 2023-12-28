import json, re
from collections import Counter
from math import log, inf
from functools import cache
from typing import List

@cache
def tokenize(text):
    return [y for y in [re.sub('[^a-z0-9]', '', x) for x in text.lower().split(" ")]  if len(y)]

def train(dataset):
    global count_of_word_by_category
    global num_data_points
    global num_data_points_in_category
    count_of_word_by_category = {}
    num_data_points = len(dataset)
    num_data_points_in_category = Counter()
    for point in dataset:
        name = point['name']
        classification = point['classification']
        num_data_points_in_category[classification] += 1
        if classification not in count_of_word_by_category:
            count_of_word_by_category[classification] = Counter()
        words = set(tokenize(point['contents']))
        for word in words:
            count_of_word_by_category[classification][word] += 1

@cache
def pr_category(category : str):
    """
    Computes Pr(category)
    """
    return 0

@cache
def pr_word_given_category(word : str, category : str, num_words_in_document : int): 
    """
    Computes Pr(word | category)
    """
    return 0

def log_pr_category_given_words(words : List[str], category : str):
    """
    Computes log(Pr(category | words))
    """
    return 0

def predict(categories, words):
    best = None
    best_likelihood = -inf
    for category in categories:
        pr = log_pr_category_given_words(words, category)
        if  pr > best_likelihood:
            best = category
            best_likelihood = pr
    return best
