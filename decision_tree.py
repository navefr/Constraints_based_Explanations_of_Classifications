__author__ = 'nfrost'

import sys
import numpy as np
import scipy

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def distance_sparse(point1, point2):
    return scipy.sparse.linalg.norm(np.array(point1) - np.array(point2))

class DecisionTree:
    def __init__(self, clf):
        self.tree = clf

        self.n_nodes = clf.tree_.node_count
        self.children_left = clf.tree_.children_left
        self.children_right = clf.tree_.children_right
        self.feature = clf.tree_.feature
        self.threshold = clf.tree_.threshold
        self.values = clf.tree_.value
        self.parent = {}
        for i in range(len(self.children_left)):
            if self.children_left[i] >= 0:
                self.parent[self.children_left[i]] = i
        for i in range(len(self.children_right)):
            if self.children_right[i] >= 0:
                self.parent[self.children_right[i]] = i

    def is_leaf(self, node_id):
        return self.children_right[node_id] == -1 and self.children_left[node_id] == -1

    def node_prediction(self, node_id):
        values = self.values[node_id][0].tolist()
        return values.index(max(values))

    def path_from_node(self, node_id, x):
        path = [node_id]
        while not self.is_leaf(node_id):
            feature = self.feature[node_id]
            threshold = self.threshold[node_id]
            if x[feature] <= threshold:
                node_id = self.children_left[node_id]
            else:
                node_id = self.children_right[node_id]
            path.append(node_id)
        return path

    def print_path(self, x):
        node_indicator = self.tree.decision_path([x])

        # Now, it's possible to get the tests that were used to predict a sample or
        # a group of samples. First, let's make it for the sample.
        node_index = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]
        print 'Path', node_index
        for node_id in node_index:
            if not self.is_leaf(node_id):
                if x[self.feature[node_id]] <= self.threshold[node_id]:
                    threshold_sign = "<="
                else:
                    threshold_sign = ">"

                print("decision id node %s : (X[%s] (= %s) %s %s)"
                      % (node_id,
                         self.feature[node_id],
                         x[self.feature[node_id]],
                         threshold_sign,
                         self.threshold[node_id]))
            else:
                print("decision id node %s : Leaf - prediction %s"
                      % (node_id, self.node_prediction(node_id)))

    def modify_point(self, x):
        x_new = []
        for value in x:
            x_new.append(value)

        node_indicator = self.tree.decision_path([x_new])
        node_index = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]]

        original_prediction = self.node_prediction(node_index[-1])

        for node_id in node_index[::-1][1:]:
            current_prediction = self.node_prediction(node_id)
            if current_prediction != original_prediction:
                feature = self.feature[node_id]
                threshold = self.threshold[node_id]

                if x_new[feature] <= threshold:
                    x_new[feature] = threshold + 1
                else:
                    x_new[feature] = threshold
                break

        path = self.path_from_node(node_id, x_new)
        if self.node_prediction(path[-1]) != original_prediction:
            return x_new
        else:
            return self.modify_point(x_new)

    def modify_to_value(self, x, value, constraints=lambda x: True, d=distance):
        min_changes = sys.maxint
        modified_point = None
        for node_id in range(self.n_nodes):
            if self.is_leaf(node_id) and self.node_prediction(node_id) == value:
                x_new = self.modify_to_match_a_leaf(x, node_id)
                changes = d(x, x_new)
                if changes < min_changes and constraints((x, x_new)):
                    min_changes = changes
                    modified_point = x_new
        return modified_point

    def modify_to_match_a_leaf(self, x, leaf):
        x_new = []
        for value in x:
            x_new.append(value)

        node_id = leaf
        while node_id in self.parent:
            parent_id = self.parent[node_id]
            feature = self.feature[parent_id]
            threshold = self.threshold[parent_id]
            if x_new[feature] <= threshold and self.children_left[parent_id] != node_id:
                x_new[feature] = threshold + 1
            elif x_new[feature] > threshold and self.children_right[parent_id] != node_id:
                x_new[feature] = threshold
            node_id = parent_id
        return x_new

    def modify_to_value_sparse(self, x, value, constraints=lambda x: True, d=distance_sparse):
        min_changes = sys.maxint
        modified_point = None
        for node_id in range(self.n_nodes):
            if self.is_leaf(node_id) and self.node_prediction(node_id) == value:
                x_new = self.modify_to_match_a_leaf_sparse(x, node_id)
                changes = d(x, x_new)
                # if changes < min_changes and constraints((x, x_new)):
                if changes < min_changes:
                    min_changes = changes
                    modified_point = x_new
        return modified_point

    def set_value_sparse(self, m, i, v):
        found = False
        for index in range(len(m.indices)):
            if m.indices[index] == i:
                m.data[index] = v
                found = True
        if not found:
            indices = np.append(m.indices, np.array(i))
            data = np.append(m.data, np.array(v))
            indices, data = zip(*sorted(zip(indices, data)))
            m.indices = np.array(indices)
            m.data = np.array(data)
        return m

    def modify_to_match_a_leaf_sparse(self, x, leaf):
        x_new = x.copy()
        node_id = leaf
        while node_id in self.parent:
            parent_id = self.parent[node_id]
            feature = self.feature[parent_id]
            threshold = self.threshold[parent_id]
            if x_new[0][0, feature] <= threshold and self.children_left[parent_id] != node_id:
                x_new = self.set_value_sparse(x_new[0], feature, threshold + 1)
            elif x_new[0][0, feature] > threshold and self.children_right[parent_id] != node_id:
                x_new = self.set_value_sparse(x_new[0], feature, threshold)
            node_id = parent_id
        return x_new