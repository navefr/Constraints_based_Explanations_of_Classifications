__author__ = 'nfrost'


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import numpy as np
from decision_tree import DecisionTree
import itertools
import random
import pickle

random.seed(12)

n_samples = 10000
n_train = 1000
n_misses = 300
n_features = 5
n_informative = 3
n_redundant = 2
n_clusters_per_class = 1
num_changes_allowed = 2

X, Y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_informative=n_informative, n_redundant=n_redundant,
                           n_clusters_per_class=n_clusters_per_class,
                           class_sep=0.5, random_state=44)

X_train = X[:n_train]
y_train = Y[:n_train]
X_test = X[n_train:]
y_test = Y[n_train:]

clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=32)

clf.fit(X_train, y_train)

print len(X_train)
print accuracy_score(y_train, clf.predict(X_train))
print len(X_test)
print accuracy_score(y_test, clf.predict(X_test))

predictions = clf.predict(X_test)

mistakes = []
corrects = []


for vector, prediction, tag in zip(X_test, predictions, y_test):
    if prediction == tag:
        corrects.append(vector)
    else:
        mistakes.append((vector, prediction, tag))
print len(mistakes), 'mistakes'
mistakes = mistakes[:n_misses]


def distance(point1, point2):
    return sum(np.abs(np.array(point1) - np.array(point2)))


def constraints(points_tuple):
    original = points_tuple[0]
    changed = points_tuple[1]

    number_changes = 0
    for o, c in zip(original, changed):
        if o != c:
            number_changes += 1

    return number_changes <= num_changes_allowed


trees = []
feature_values = {}
for tree in clf.estimators_:
    dt = DecisionTree(tree)
    trees.append(dt)
    for feature, value in zip(dt.feature, dt.threshold):
        if feature >= 0:
            if feature not in feature_values:
                feature_values[feature] = set()
            feature_values[feature].add(value)

with open('rf_project_opt_results.p', 'rb') as f:
    running_results = pickle.load(f)

max_beta = np.max(running_results.keys())

easy_case_indices = []
for i in range(len(running_results[max_beta])):
    case = running_results[max_beta][i]
    has_optimal = case['optimal'] is not None
    is_optimized_found = case['optimized'][1]
    if has_optimal and is_optimized_found:
        easy_case_indices.append(i)

for beta in running_results:
    beta_results = running_results[beta]
    cnt = 0
    optimal_d_from_p = 0
    optimized_d_from_p = 0
    d_from_o = 0
    for i in easy_case_indices:
        p = np.array(beta_results[i]['mistake'][0])
        optimal = beta_results[i]['optimal']
        optimized = beta_results[i]['optimized'][0]
        cnt += 1
        optimal_d_from_p += distance(p, optimal)
        optimized_d_from_p += distance(p, optimized)
        d_from_o += distance(optimal, optimized)
    print(beta, optimal_d_from_p / cnt, optimized_d_from_p / cnt, d_from_o / cnt)