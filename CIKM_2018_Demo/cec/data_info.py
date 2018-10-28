__author__ = 'nfrost'

import numpy as np
import pandas as pd
import collections


class DataInfo:

    def __init__(self, df, col_names):
        col_num = np.array(df.col_names)[np.array(list(map(int, df.columns_by_type())))]
        col_num_bins = {}
        for col in col_num:
            success = False
            q = 20
            while not success:
                try:
                    col_num_bins[col] = pd.qcut(df.as_data_frame()[col], q, retbins=True, duplicates='drop')[1]
                    success = True
                except:
                    if q > 2:
                        q = int(q / 2)
                    else:
                        raise Exception('Fail to bin ' + col)

        col_cat = list(map(lambda x: col_names[int(x)], df[col_names].columns_by_type("categorical")))
        col_cat_dist = {}

        for col in col_cat:
            values = collections.Counter(df.as_data_frame()[col]).most_common()
            s = sum(map(lambda x: x[1], values))
            dist = {}
            for value in values:
                dist[value[0]] = value[1] / s
            col_cat_dist[col] = dist

        self.col_names = col_names
        self.col_num_bins = col_num_bins
        self.col_cat_dist = col_cat_dist
        self.categorical_features = list(map(int, df[col_names].columns_by_type("categorical")))

    def get_col_names(self):
        return self.col_names

    def get_categorical_features(self):
        return self.categorical_features

    def get_col_num_bins(self):
        return self.col_num_bins

    def get_col_cat_dist(self):
        return self.col_cat_dist

    def get_bin_range(self, col, value):
        bins = self.col_num_bins[col]
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                break
        return bins[i], bins[i + 1]
