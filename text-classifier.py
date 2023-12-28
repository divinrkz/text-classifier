import json, re
from collections import Counter
import bayes

TRAIN = 'data/train.json'
VALIDATE = 'data/validate.json'

train = json.loads(open(TRAIN).read())
validate = json.loads(open(VALIDATE).read())

def test(dataset, categories):
    answers = dict([x.split(" ") for x in open(dataset + "_validate.txt").read().split("\n")[:-1]])

    bayes.train(train[dataset])

    correct_by_category = Counter()
    incorrect_by_category = Counter()

    for point in validate[dataset]:
        words = set(bayes.tokenize(point['contents']))
        prediction = bayes.predict(categories, words)
        answer = answers[point['name']]
        if prediction == answer:
            correct_by_category[answer] += 1
        else:
            incorrect_by_category[answer] += 1

    print(correct_by_category)
    print(incorrect_by_category)

test('tweets', set(['positive', 'negative']))
test('emails', set(['spam', 'ham']))

