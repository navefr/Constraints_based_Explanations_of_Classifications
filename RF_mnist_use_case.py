__author__ = 'nfrost'

import random
import os
from keras.datasets import mnist

random.seed(43)
output_directory = 'output/mnist'
run_type = 'only_deletion'

(images_train, y_train), (images_test, y_test) = mnist.load_data()

n_samples = len(images_train)
X_train = images_train.reshape((n_samples, -1))
n_samples = len(images_test)
X_test = images_test.reshape((n_samples, -1))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=52)

clf.fit(X_train, y_train)

print(len(X_train))
print(accuracy_score(y_train, clf.predict(X_train)))
print(len(X_test))
print(accuracy_score(y_test, clf.predict(X_test)))

predictions = clf.predict(X_test)

mistakes = []
corrects = []

wanted_tag = 4
predicted_tag = 9

for vector, prediction, tag in zip(X_test, predictions, y_test):
    if not prediction == tag and tag == wanted_tag and prediction == predicted_tag:
        mistakes.append((vector, prediction, tag))
    if prediction == tag and tag == wanted_tag:
        corrects.append(vector)

from skimage import color
import numpy as np
import matplotlib.pyplot as plt
from decision_tree import DecisionTree


def plot_modified_and_diff(clf, plt, vector, modified_vector, prefix):
    modified_img = np.array(modified_vector).reshape(28, 28)
    modified_prediction = clf.predict([modified_vector])[0]
    modified_proba = round(clf.predict_proba([modified_vector])[0][modified_prediction], 3)

    plt.imshow(modified_img, cmap='gray')
    plt.axis('off')
    image_name = '_'.join([prefix, 'modified', str(actual_tag), 'tagged', str(modified_prediction), 'p', str(modified_proba)])
    plt.savefig(os.path.join(output_directory, image_name + '.jpg'))
    plt.clf()

    diff = []
    changes = 0
    for value, modified_value in zip(vector, modified_vector):
        if value == modified_value:
            diff.append((1, 1, 1))
        else:
            d = abs(value - modified_value) / 2
            changes += 1
            if modified_value > value:
                diff.append((0, 0, (255.0 / 2 + d) / 255.0))
            else:
                diff.append(((255.0 / 2 + d) / 255.0, 0, 0))
    diff_img = np.array(diff).reshape((28, 28, 3))
    plt.imshow(diff_img, cmap="gray")
    plt.axis('off')
    image_name = '_'.join([prefix, 'diff', str(changes), 'pixels'])
    plt.savefig(os.path.join(output_directory, image_name + '.jpg'))
    plt.clf()


def get_diffs(vector, modified_vector):
    diff = []
    for value, modified_value in zip(vector, modified_vector):
        if value == modified_value:
            diff.append((1, 1, 1))
        else:
            d = abs(value - modified_value) / 2
            if modified_value > value:
                diff.append((0, 0, (255.0 / 2 + d) / 255.0))
            else:
                diff.append(((255.0 / 2 + d) / 255.0, 0, 0))

    return np.array(diff)


def distance(point1, point2):
    return sum(np.abs(np.array(point1) - np.array(point2)))


def constraints(points_tuple):
    original = points_tuple[0]
    changed = points_tuple[1]

    for i in range(len(original)):
        if original[i] < changed[i]:
            return False
    return True


def multi_step_algorithm(clf, trees, vector, original_probas, actual_tag):
    modified_vector = vector
    prev_proba = original_probas[actual_tag]
    prev_proba_improve = 0
    for i in range(100):
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
        proba_improve = max_proba - prev_proba
        if proba_improve < 0.25 * prev_proba_improve:
            break
        else:
            prev_proba = max_proba
            prev_proba_improve = proba_improve
            modified_vector = modified_vectors[max_index]
    return modified_vector

trees = []
for tree in clf.estimators_:
    trees.append(DecisionTree(tree))


average_correct = np.zeros(28 * 28)
for vector in corrects:
    average_correct += np.array(vector)

average_correct /= len(corrects)
plt.imshow(average_correct.reshape(28, 28), cmap='gray')
plt.axis('off')
image_name = '_'.join([run_type, 'avg', str(wanted_tag), 'tagged', str(wanted_tag)])
plt.savefig(os.path.join(output_directory, image_name + '.jpg'))
plt.clf()


for i in range(10):
    mistake = mistakes[i]
    vector = np.array(mistake[0])
    model_prediction = mistake[1]
    actual_tag = mistake[2]
    original_probas = clf.predict_proba([vector])[0]

    prefix = str(i) + '_' + run_type

    print()
    print('Model Prediction', model_prediction)
    print('Actual Tag', actual_tag)

    image = vector.reshape((28, 28))
    rgb_image = color.gray2rgb(image)

    plt.imshow(rgb_image, cmap='gray')
    plt.axis('off')
    image_name = '_'.join([prefix, 'original', str(actual_tag), 'tagged', str(model_prediction), 'p', str(round(original_probas[model_prediction], 3))])
    plt.savefig(os.path.join(output_directory, image_name + '.jpg'))
    plt.clf()

    modified_vector = multi_step_algorithm(clf, trees, vector, original_probas, actual_tag)
    plot_modified_and_diff(clf, plt, vector, modified_vector, prefix)


print('Mistakes', len(mistakes))
total_diffs_actual = np.zeros((28 * 28, 3))

average_img = np.zeros(28 * 28)

for mistake in mistakes:
    vector = np.array(mistake[0])
    model_prediction = mistake[1]
    actual_tag = mistake[2]
    original_probas = clf.predict_proba([vector])[0]

    image = vector.reshape(28, 28)

    diffs_actual = get_diffs(vector, multi_step_algorithm(clf, trees, vector, original_probas, actual_tag))

    total_diffs_actual += diffs_actual

    average_img += vector

total_diffs_actual /= len(mistakes)
average_img /= len(mistakes)


plt.imshow(average_img.reshape(28, 28), cmap='gray')
plt.axis('off')
image_name = '_'.join([run_type, 'avg', str(actual_tag), 'tagged', str(model_prediction)])
plt.savefig(os.path.join(output_directory, image_name + '.jpg'))
plt.clf()

plt.imshow(total_diffs_actual.reshape((28, 28, 3)), cmap='gray')
plt.axis('off')
image_name = '_'.join([run_type, 'avg_changes', str(actual_tag), 'tagged', str(actual_tag)])
plt.savefig(os.path.join(output_directory, image_name + '.jpg'))
plt.clf()
