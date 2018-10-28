__author__ = 'nfrost'

import h2o
import subprocess
import pygraphviz as pgv
import numpy as np
import pandas as pd


def distance(x, y, col_names):
    d = 0
    for col in col_names:
        if (x[col] != y[col]).all():
            d += 1
    return d


class DecisionTreeH2O:
    def __init__(self, drf, tree_id, data_info,
                 h2o_jar_path='/Users/nfrost/Downloads/h2o-3.18.0.4/h2o.jar',
                 mojo_full_path='trees/tree_mojo.zip', gv_file_path='trees/tree_gv.gv',
                 seed=0):
        drf.download_mojo(mojo_full_path)
        result = subprocess.call(
            ["java", "-cp", h2o_jar_path, "hex.genmodel.tools.PrintMojo", "--tree", str(tree_id), "--levels", str(500), "-i", mojo_full_path,
             "-o", gv_file_path], shell=False)
        result = subprocess.call(["ls", gv_file_path], shell=False)
        if result is 0:
            self.seed = seed
            self.G = pgv.AGraph(gv_file_path)
            self.data_info = data_info
            self.feature_name_index = {}
            for index in range(len(data_info.get_col_names())):
                self.feature_name_index[data_info.get_col_names()[index]] = index
        else:
            print("Error: Graphviz file " + gv_file_path + " could not be generated.")

    def is_leaf(self, node_id):
        return self.G.out_degree(node_id) == 0

    def leaf_prediction(self, node_id):
        return float(self.G.get_node(node_id).attr.get('label'))

    def get_node_feature(self, node_id):
        return self.G.get_node(node_id).attr.get('label')

    def modify_to_tag(self, x, wanted_tag, error, d=distance, constraints=lambda x: True, project=lambda x: x):
        min_changes = float("inf")
        modified_point = None
        for node_id in self.G.nodes():
            if self.is_leaf(node_id) and abs(wanted_tag - self.leaf_prediction(node_id)) <= error:
                x_new = self.modify_to_match_a_leaf(x, node_id)
                x_new = project(x_new)
                changes = d(x, x_new, self.data_info.get_col_names())
                if 0 < changes < min_changes and constraints((x, x_new, wanted_tag)):
                    min_changes = changes
                    modified_point = x_new
        return modified_point

    def modify_to_match_a_leaf(self, x, leaf):
        x_new = x.copy()

        node_id = leaf
        while self.G.in_degree(node_id) > 0:
            in_neighbors = self.G.in_neighbors(node_id)
            assert len(in_neighbors) == 1
            parent_id = in_neighbors[0]
            parent_feature = self.get_node_feature(parent_id)
            edge = self.G.get_edge(parent_id, node_id)

            is_numeric = ' < ' in parent_feature

            if is_numeric:
                parent_feature, threshold = parent_feature.split(' < ')
                parent_feature_index = self.feature_name_index[parent_feature]
                threshold = float(threshold)
                is_condition_less_than = '<' in edge.attr.get('label')
                if is_condition_less_than:
                    if (x_new[parent_feature] >= threshold).all():
                        x_new[parent_feature] = int(threshold - 1)
                    else:
                        pass
                else:
                    if (x_new[parent_feature] >= threshold).all():
                        pass
                    else:
                        x_new[parent_feature] = int(threshold + 1)
            else:
                parent_feature_index = self.feature_name_index[parent_feature]
                wanted_values = edge.attr.get('label').split('\\n')
                if not (x_new[parent_feature].isin(wanted_values)).all():
                    if '' in wanted_values:
                        wanted_values.remove('')
                    if '[NA]' in wanted_values and len(wanted_values) > 1:
                        wanted_values.remove('[NA]')

                    p = []
                    for value in wanted_values:
                        if value in self.data_info.get_col_cat_dist()[parent_feature]:
                            p.append(self.data_info.get_col_cat_dist()[parent_feature][value])
                        else:
                            p.append(0.01)
                    p = np.array(p)
                    p = p / sum(p)

                    np.random.seed(self.seed)
                    x_new[parent_feature] = np.random.choice(wanted_values, p=p)
            node_id = parent_id
        return x_new


def multi_step_algorithm(clf, trees, vector, col_names, tag, constraints=lambda x: True, project=lambda x: x):
    error = 0.4
    value = 0.0 if tag == 'yes' else 1.0
    modified_vector = vector.copy()
    prev_proba = clf.predict(h2o.H2OFrame(vector))[tag].as_data_frame().values[0][0]
    prev_proba_improve = 0
    for i in range(10):
        modified_vectors = [modified_vector]
        for decision_tree in trees:
            candidate_vector = decision_tree.modify_to_tag(modified_vector, value, error, d=distance,
                                                           constraints=constraints, project=project)
            if candidate_vector is not None:
                modified_vectors.append(candidate_vector)
        modified_vectors_frame = h2o.H2OFrame(pd.concat(modified_vectors))
        predictions = clf.predict(modified_vectors_frame)
        modified_vectors_frame['prediction'] = predictions[tag]

        max_proba = modified_vectors_frame['prediction'].max()
        proba_improve = max_proba - prev_proba
        if proba_improve > 0.0:
            is_significant_improve = proba_improve > 0.01 * prev_proba_improve
            prev_proba = max_proba
            prev_proba_improve = proba_improve
            modified_vector = \
            modified_vectors_frame[modified_vectors_frame['prediction'] == max_proba][col_names].as_data_frame().iloc[
                [0]].copy()
            if not is_significant_improve:
                break
        else:
            break
    return modified_vector


def get_find_error_changes(model, trees, col_names, wanted_tag, constraints=lambda x: True, project=lambda x: x):
    def find_error_changes(vector):
        vector = vector.to_frame().transpose()[col_names]
        ans = pd.Series(
            multi_step_algorithm(model, trees, vector, col_names, wanted_tag, constraints=constraints, project=project).values[
                0], col_names)
        return ans

    return find_error_changes


class ConstraintsTabularExplainer:

    def __init__(self, data_info, ntrees, constraints=lambda x: True, project=lambda x: x):
        self.data_info = data_info
        self.constraints = constraints
        self.project = project
        self.ntrees = ntrees

    def get_model_trees(self, model):
        trees = []
        for i in range(self.ntrees):
            trees.append(DecisionTreeH2O(model, i, self.data_info, gv_file_path='trees/tree_' + str(i) + '_gv.gv'))
        return trees

    def get_changes(self, model, valid, prediction='no', wanted_tag='yes', n_exp=1, seed=None):
        valid['predict'] = model.predict(valid)['predict']
        errors = valid[(valid['y'] == wanted_tag) & (valid['predict'] == prediction)].as_data_frame()

        if len(errors) > n_exp:
            errors = errors[errors['contact'] == 'unknown']
            errors_to_explain = errors.sample(n=n_exp, random_state=seed)
        else:
            errors_to_explain = errors
        if len(errors) > 0:
            trees = self.get_model_trees(model)
            find_error_changes = get_find_error_changes(model, trees, self.data_info.get_col_names(), wanted_tag,
                                                        constraints=self.constraints, project=self.project)

            changes = errors_to_explain.apply(find_error_changes, axis=1)
            changes['y'] = errors_to_explain['y']
            changes['predict'] = model.predict(h2o.H2OFrame(changes))['predict'].as_data_frame().values[0]

            df_all = pd.concat([errors_to_explain, changes], keys=['Error', 'Changed'])

            def highlight_diff(data, color='yellow'):
                attr = 'background-color: {}'.format('yellow') 
                error_highlight = pd.DataFrame(np.where(data.xs('Error') != data.xs('Changed'), attr, ''),
                                               index=data.xs('Error').index, columns=data.xs('Error').columns)    
                changed_highlight = pd.DataFrame(np.where(data.xs('Error') != data.xs('Changed'), attr, ''),
                                                 index=data.xs('Changed').index, columns=data.xs('Changed').columns)
                return pd.concat([error_highlight, changed_highlight], keys=['Error', 'Changed'])
            
            return df_all.style.apply(highlight_diff, axis=None)
        else:
            return None, None

    def get_additional_data(self, model, train, valid, out, step, prediction='no', wanted_tag='yes', n_exp=10,
                            top_n_exp=5, seed=1):
        n = int(step * len(out))

        errors_to_explain, changes = self.get_changes(model, valid, prediction, wanted_tag, n_exp, seed)

        if errors_to_explain is not None:
            ne_stacked = (changes != errors_to_explain[self.data_info.get_col_names()]).stack()
            changed = ne_stacked[ne_stacked]
            changed.index.names = ['id', 'col']

            difference_locations = np.where(changes != errors_to_explain[self.data_info.get_col_names()])
            changed_to = changes.values[difference_locations]
            changed_from = errors_to_explain[self.data_info.get_col_names()].values[difference_locations]
            diff = pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)

            g = diff.groupby(['col', 'from', 'to'])
            top_explanations = sorted(g, key=lambda x: len(x[1]), reverse=True)[:top_n_exp]

            exp_sizes = [len(exp[1]) for exp in top_explanations]
            sum_exp_sizes = sum(exp_sizes)
            exp_proportions = [size / sum_exp_sizes for size in exp_sizes]

            in_train = None
            for explanation, exp_proportion in zip(top_explanations, exp_proportions):
                feature = explanation[0][0]
                from_value = explanation[0][1]
                to_value = explanation[0][2]

                if feature in self.data_info.get_col_num_bins():
                    from_start, from_end = self.data_info.get_bin_range(feature, from_value)
                    to_start, to_end = self.data_info.get_bin_range(feature, to_value)
                    data_from = out[
                        (out[feature] >= from_start) & (out[feature] <= from_end) & (out['y'] == wanted_tag)]
                    data_to = out[(out[feature] >= to_start) & (out[feature] <= to_end) & (out['y'] == wanted_tag)]
                else:
                    data_from = out[(out[feature] == from_value) & (out['y'] == wanted_tag)]
                    data_to = out[(out[feature] == to_value) & (out['y'] == wanted_tag)]

                exp_n = n * exp_proportion
                ratio_from = 0.9999 if len(data_from) == 0 else exp_n / (2 * len(data_from))
                ratio_from = 0.9999 if ratio_from >= 1.0 else ratio_from
                data_from, _ = data_from.split_frame(ratios=[ratio_from])

                ratio_to = 0.9999 if len(data_to) == 0 else exp_n / (2 * len(data_to))
                ratio_to = 0.9999 if ratio_to >= 1.0 else ratio_to
                data_to, _ = data_to.split_frame(ratios=[ratio_to])

                if in_train is None:
                    in_train = data_from
                else:
                    in_train = in_train.rbind(data_from)
                in_train = in_train.rbind(data_to)

            return in_train
        else:
            return None
