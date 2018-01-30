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


def find_optimal(clf, vector, feature_values, num_changes_allowed):
    all_combinations = []
    for changes in range(1, num_changes_allowed + 1):
        for combination in itertools.combinations(range(len(vector)), changes):
            values = list(itertools.product(*[feature_values[i] for i in combination]))
            for i in range(len(values)):
                vector_copy = vector.copy()
                for index, value in zip(combination, values[i]):
                    vector_copy[index] = value
                all_combinations.append(vector_copy)
    all_combinations_probas = clf.predict_proba(all_combinations)

    optimal_combination = None
    closest_d = None
    for combination, proba in zip(all_combinations, all_combinations_probas):
        if constraints((vector, combination)) and proba[actual_tag] > beta:
            d = distance(vector, combination)
            if closest_d is None or d < closest_d:
                closest_d = d
                optimal_combination = combination
    return optimal_combination


def optimized_algorithm(clf, trees, vector, actual_tag, beta, constraints, max_iters):
    modified_vector = vector

    max_proba = 0
    for i in range(max_iters):
        modified_vectors = [vector]
        for decision_tree in trees:
            candidate_vector = decision_tree.modify_to_value(modified_vector, actual_tag, constraints=constraints, d=distance)
            if candidate_vector is not None:
                modified_vectors.append(candidate_vector)
        probas = clf.predict_proba(modified_vectors)
        max_proba = 0
        max_index = None
        for j in range(len(probas)):
            if probas[j][actual_tag] > max_proba:
                max_proba = probas[j][actual_tag]
                max_index = j
        modified_vector = modified_vectors[max_index]
        if max_proba > beta:
            break
    return modified_vector, max_proba >= beta, i + 1


def non_optimized_algorithm(clf, trees, vector, actual_tag, beta, constraints, max_iters):
    modified_vector = vector

    max_proba = 0
    for i in range(max_iters):
        modified_vectors = [vector]
        for decision_tree in trees:
            candidate_vector = decision_tree.modify_to_value(modified_vector, actual_tag, constraints=lambda x: True, d=distance)
            if candidate_vector is not None:
                modified_vectors.append(candidate_vector)
        probas = clf.predict_proba(modified_vectors)
        max_proba = 0
        max_index = None
        for j in range(len(probas)):
            if probas[j][actual_tag] > max_proba:
                max_proba = probas[j][actual_tag]
                max_index = j
        modified_vector = modified_vectors[max_index]
        if not constraints((vector, modified_vector)):
            indexed_changed = []
            indexed_d = []
            for i in range(len(vector)):
                if vector[i] != modified_vector[i]:
                    indexed_changed.append(i)
                    indexed_d.append(abs(modified_vector[i] - vector[i]))
            indexed_d, indexed_changed = zip(*sorted(zip(indexed_d, indexed_changed)))
            for i in range(len(indexed_changed) - num_changes_allowed):
                modified_vector[indexed_changed[i]] = vector[indexed_changed[i]]
            max_proba = clf.predict_proba([modified_vector])[0][actual_tag]
        if max_proba > beta:
            break
    return modified_vector, max_proba >= beta, i + 1


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

print '\t'.join(['Beta', 'Examples', 'Examples_with_optimal', 'avg_d_optimal', 'avg_d_optimize',  'avg_d_non_optimize', 'avg_d_optimize_to_optimal', 'avg_d_non_optimize_to_optimal', 'avg_optimize_iters', 'avg_non_optimize_iters', 'pct_optimize_found', 'pct_non_optimize_found', 'cnt_non_opt_found_something_else', 'cnt_mismatches'])

running_results = {}

for beta in np.arange(0.55, 0.975, 0.025):
    has_optimal = 0
    total = 0
    sum_d_optimal = 0
    sum_d_optimize = 0
    cnt_optimize_found = 0
    sum_d_non_optimize = 0
    cnt_non_optimize_found = 0
    sum_d_optimize_to_optimal = 0
    sum_d_non_optimize_to_optimal = 0
    sum_optimize_iters = 0
    sum_non_optimize_iters = 0
    cnt_non_opt_found_something_else = 0
    cnt_mismatches = 0

    running_results[beta] = []

    for mistake in mistakes:
        vector = np.array(mistake[0])
        model_prediction = mistake[1]
        actual_tag = mistake[2]
        original_probas = clf.predict_proba([vector])[0]

        optimized_vector, is_optimized_found, optimized_iters = optimized_algorithm(clf, trees, vector, actual_tag, beta, constraints, 30)
        non_optimized_vector, is_non_optimized_found, non_optimized_iters = non_optimized_algorithm(clf, trees, vector, actual_tag, beta, constraints, 30)
        optimal_combination = find_optimal(clf, vector, feature_values, num_changes_allowed)

        if optimal_combination is not None:
        # if True:
            has_optimal += 1
            sum_d_optimal += distance(vector, optimal_combination)
            if is_optimized_found:
                cnt_optimize_found += 1
                sum_d_optimize += distance(vector, optimized_vector)
                sum_d_optimize_to_optimal += distance(optimal_combination, optimized_vector)
                sum_optimize_iters += optimized_iters
            if is_non_optimized_found:
                cnt_non_optimize_found += 1
                sum_d_non_optimize += distance(vector, non_optimized_vector)
                sum_d_non_optimize_to_optimal += distance(optimal_combination, non_optimized_vector)
                sum_non_optimize_iters += non_optimized_iters
            if is_non_optimized_found and not is_optimized_found:
                cnt_non_opt_found_something_else += 1
            if is_non_optimized_found and is_optimized_found and optimized_vector != non_optimized_vector:
                cnt_mismatches += 1

        total += 1

        running_results[beta].append({'mistake': mistake,
                                      'optimal': optimal_combination,
                                      'optimized': (optimized_vector, is_optimized_found, optimized_iters),
                                      'non_optimized': (non_optimized_vector, is_non_optimized_found, non_optimized_iters)})

    to_print = [str(beta),
                str(total),
                str(round(100 * float(has_optimal) / total, 3)) + '%',
                str(round(float(sum_d_optimal) / has_optimal, 3)),
                str(round(float(sum_d_optimize) / cnt_optimize_found, 3)),
                str(round(float(sum_d_non_optimize) / cnt_non_optimize_found, 3)),
                str(round(float(sum_d_optimize_to_optimal) / cnt_optimize_found, 3)),
                str(round(float(sum_d_non_optimize_to_optimal) / cnt_non_optimize_found, 3)),
                str(round(float(sum_optimize_iters) / cnt_optimize_found, 3)),
                str(round(float(sum_non_optimize_iters) / cnt_non_optimize_found, 3)),
                str(round(100 * float(cnt_optimize_found) / has_optimal, 3)) + '%',
                str(round(100 * float(cnt_non_optimize_found) / has_optimal, 3)) + '%',
                str(cnt_non_opt_found_something_else),
                str(cnt_mismatches)]
    print '\t'.join(to_print)

with open('rf_project_opt_results.p', 'wb') as f:
    pickle.dump(running_results, f)