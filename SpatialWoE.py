__author__ = 'Denis Surzhko'

import numpy as np
import abc
import matplotlib.pyplot as plt
from sklearn import tree
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import cross_val_score

class SpatialWoEBase(metaclass=abc.ABCMeta):

    def __init__(self):
        self.x = None  # predictors
        self.y = None  # target variable (n x 1)
        self.c_x = None  # regions for each x dim
        self.region = None # region for each observation
        self.woe = None  # woe values for each observation
        self.region_woe = None # dict region: woe value
        self._dict_reg_convert = None  # dictionary region: list of leafs in the tree

    def fit(self, x, y):
        if not isinstance(x, np.ndarray):
            raise Exception("X should be numpy ndarray")
        if not isinstance(y, np.ndarray):
            raise Exception("y should be numpy ndarray")
        if y.shape[1] != 1:
            raise Exception("y should have (n, 1) shape")
        if y.shape[0] != x.shape[0]:
            raise Exception("X and y should have the same number of observations")

        self.x = x.copy()
        self.y = y.copy()

        self._split_regions()
        self._calc_woe()

    def _calc_woe(self):
        t_bad = np.maximum(self.y.sum(), 0.5)
        t_good = np.maximum(self.y.shape[0] - self.y.sum(), 0.5)
        self.region_woe = {}
        self.woe = np.zeros((self.region.shape[0], 1))
        for region in self._dict_reg_convert.keys():
            sub_area = self.y[self.region == region]
            bad = sub_area.sum()
            good = sub_area.shape[0] - bad
            self.region_woe[region] = self._bucket_woe(bad, good) + np.log(t_bad / t_good)
            self.woe[self.region == region] = self.region_woe[region]

    def transform(self, x):
        regions = self.predict(x)
        woe = np.zeros((x.shape[0], 1))
        for region in np.unique(regions):
            woe[regions == region] = self.region_woe[region]
        return woe

    def plot(self):
        fig = plt.figure()
        woe_norm = np.array([self.region_woe[w] for w in sorted(self.region_woe.keys())])
        woe_norm -= woe_norm.min()
        woe_norm /= woe_norm.max()
        colors = ','.join(["#{:02x}".format(int(np.abs(w)*255)) + "0000" for w in woe_norm])
        plot_decision_regions(self.x, self.region.flatten(), clf=self, legend=2, colors = colors)
        return fig

    def merge(self, merge_pair):
        keep = np.min(merge_pair)
        remove = np.max(merge_pair)
        max_region = np.max(list(self.region_woe.keys()))

        self._dict_reg_convert[keep].extend(self._dict_reg_convert[remove])

        for i in range(remove, max_region):
            self._dict_reg_convert[i] = self._dict_reg_convert[i+1]

        del self._dict_reg_convert[max_region]

        self.region = self.predict(self.x)
        self._calc_woe()


    @staticmethod
    def _bucket_woe(bad, good):
        bad = 0.5 if bad == 0 else bad
        good = 0.5 if good == 0 else good
        return np.log(good / bad)

    @staticmethod
    def _replace_values_from_dict(column, new_values):
        new_column = column.copy()
        for key, values in new_values.items():
            for v in values:
                np.place(new_column, column == v, key)
        return new_column

    @abc.abstractmethod
    def _split_regions(self):
        return

    @abc.abstractmethod
    def predict(self, x):
        return


class QuantileSpatialWoE(SpatialWoEBase):

    def __init__(self, qnt_num=(1, 2), predefined_borders=None):
        super().__init__()
        self._qnt_num = qnt_num  # number of quartiles for each x dim (tuple)
        if predefined_borders is None:
            self._predefined_borders = False  # Predefined borders flag
            self.borders = None  # Borders used to split the sample
        else:
            self._predefined_borders = True  # Predefined borders flag
            self.borders = predefined_borders  # list of tuples with predefined split borders
        self.__REGION_SHIFT = 1000  # shift for next dim

    def _split_regions(self):
        self.c_x = None
        regions_raw = None
        if not self._predefined_borders:
            self.borders = []
        for c_num in range(self.x.shape[1]):
            column = self.x[:, [c_num]]
            qnt_num = self._qnt_num[c_num]
            if self._predefined_borders:
                c_x = np.digitize(column, self.borders[c_num].flatten()).reshape(-1, 1)
            else:
                c_x, border = self.__qnt_column(column, qnt_num)
                self.borders.append(border)
            if self.c_x is None:
                self.c_x = c_x
                regions_raw = c_x
            else:
                self.c_x = np.hstack((self.c_x, c_x))
                regions_raw += c_x * c_num * self.__REGION_SHIFT
        self._dict_reg_convert = {new_reg: [old_reg] for (new_reg, old_reg) \
                                  in enumerate(sorted(np.unique(regions_raw)))}
        self.region = self._replace_values_from_dict(regions_raw, self._dict_reg_convert)

    def predict(self, x):
        for c_num in range(x.shape[1]):
            column = x[:, [c_num]]
            border = self.borders[c_num]
            column_cut = np.digitize(column, border.flatten())
            if c_num == 0:
                regions_raw = column_cut
            else:
                regions_raw += column_cut * c_num * self.__REGION_SHIFT
        return self._replace_values_from_dict(regions_raw, self._dict_reg_convert)


    @staticmethod
    def __qnt_column(column, qnt_num):
        qnt_range = np.linspace(0, 100, qnt_num + 2)[1: -1]
        borders = np.percentile(column, qnt_range, axis=0)
        column_cut = np.digitize(column, borders.flatten())
        return (column_cut.reshape(-1, 1), borders.reshape(-1, 1))


class OptimizedSpatialWoE(SpatialWoEBase):

    def __init__(self, tree_args={'max_depth': 3}, cv=3, cv_scoring=None, t_type='b'):
        super().__init__()
        self.tree_args = tree_args # dict with optional args for decision tree
        self.tree = None  # decision tree for optimize approach
        self.cv = cv  # number of cv folds for cross-validation (if none - tree is fitted without cv search)
        self.cv_scoring = cv_scoring # scorer for cross-validation
        self.t_type = t_type # type of target variable (binary 'b' or continuous 'c'

    def _split_regions(self):
        tree_args = {} if self.tree_args is None else self.tree_args.copy()

        if self.t_type == 'b':
            right_tree = tree.DecisionTreeClassifier
        else:
            right_tree = tree.DecisionTreeRegressor

        if self.cv is not None:
            start = 1
            possible_depth = int(np.log2(self.x.shape[0]))
            m_depth = self.tree_args.get('max_depth', possible_depth)
            cv_scores = []
            for i in range(start, m_depth):
                tree_args['max_depth'] = i
                tr = right_tree(**tree_args)
                scores = cross_val_score(tr, self.x, self.y, cv=self.cv, scoring=self.cv_scoring)
                cv_scores.append(scores.mean())
            best = np.argmax(cv_scores) + start
            tree_args['max_depth'] = best
        self.tree = right_tree(**tree_args)
        self.tree.fit(self.x, self.y)

        leafs = (self.tree.tree_.feature == -2).nonzero()[0]
        self._dict_reg_convert = {new_reg: [old_reg] for (new_reg, old_reg) in enumerate(sorted(leafs))}
        self.region = self.predict(self.x)

    def predict(self, x):
        regions_raw = self.tree.apply(x).reshape(-1, 1)
        return self._replace_values_from_dict(regions_raw, self._dict_reg_convert)

# Examples
if __name__ == "__main__":
    # Data generation
    N = 100
    np.random.seed(1)
    x = np.random.randn(N, 2)
    y = np.where(np.random.rand(N, 1) / 10 + np.sum(x, axis=1, keepdims=True) < 0, 1, 0)
    y_c = np.random.rand(N, 1) / 10 + np.sum(x, axis=1, keepdims=True)
    y_c -= y_c.min()
    y_c /= y_c.max()


    # Optimized WoE test cases
    sw = OptimizedSpatialWoE()
    sw.fit(x, y)
    plt.show(sw.plot())
    sw.merge((1, 2))
    sw.merge((1, 2))
    plt.show(sw.plot())

    # Optimized WoE continuous test cases
    sw = OptimizedSpatialWoE(tree_args={'max_depth': 6}, t_type='c')
    sw.fit(x, y_c)
    plt.show(sw.plot())

    # Quantile WoE test cases
    sw = QuantileSpatialWoE()
    sw.fit(x, y)
    plt.show(sw.plot())
    sw.merge((1, 2))
    sw.merge((1, 2))
    plt.show(sw.plot())
    new_borders = sw.borders.copy()
    new_borders[0] = sw.borders[0] + 1
    sw = QuantileSpatialWoE(predefined_borders=new_borders)
    sw.fit(x, y)
    plt.show(sw.plot())
