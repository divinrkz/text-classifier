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
    return (num_data_points_in_category[category] / num_data_points)

@cache
def pr_word_given_category(word : str, category : str, num_words_in_document : int): 
    """
    Computes Pr(word | category)
    
    """
    return ((count_of_word_by_category[category][word] + 1) / (num_data_points_in_category[category] + num_words_in_document))

def log_pr_category_given_words(words : List[str], category : str):
    """
    Computes log(Pr(category | words))
    """
    pr_word_given_category_sum = 0
    for word in words:
        pr_word_given_category_sum += log(pr_word_given_category(word, category, len(words)))
    return pr_word_given_category_sum + log(pr_category(category))

def predict(categories, words):
    best = None
    best_likelihood = -inf
    for category in categories:
        pr = log_pr_category_given_words(words, category)
        if  pr > best_likelihood:
            best = category
            best_likelihood = pr
    return best
