import numpy as np
import pandas as pd
import collections

left_split_type = '<'
right_split_type = '>='


class DecisionTree:

    def __init__(self, max_depht):
        self.max_depht = max_depht
        self.root = None

    def train(self, x, y):
        verify(x, y)
        self.root = self.Node(question=None, depht=0)
        self._append_nodes(self.root, x, y)

    def _append_nodes(self, parent, x, y):
        if self.max_depht <= parent.depht:
            return

        feature, value = self.best_split(x, y)
        left_question = str(feature) + left_split_type + str(value)
        right_question = str(feature) + right_split_type + str(value)
        parent.left_child = self.Node(left_question, parent.depht + 1)
        parent.right_child = self.Node(right_question, parent.depht + 1)
        x_left, y_left, x_right, y_right = split_data(x, y, left_question)
        self._append_nodes(parent.left_child, x_left, y_left)
        self._append_nodes(parent.right_child, x_right, y_right)

    class Node:
        def __init__(self, question=None, depht=None):
            self.left_child = None
            self.right_child = None
            self.question = question
            self.depht = depht


def entropy(y):
    """ Entropy measures the degree of uncertainty,
     impurity or disorder of a random variable

    :param target: (list) target values
    :return: (float)  entropy of the array
    """

    counter = collections.Counter(y)
    p = np.array(list(counter.values())) / len(y)
    return -np.sum(p * np.log2(p))


def verify(x, y):
    if (isinstance(x, pd.DataFrame) or isinstance(x, pd.DataFrame)) and (
            isinstance(x, pd.DataFrame) or isinstance(y, pd.Series)):
        pass
    else:
        raise TypeError("x and y must be Pandas DataFrame")


def best_split(x, y):
    k = x.apply(best_split_value, y=y.squeeze(), split_type=left_split_type)
    gini_values = k.iloc[1]  # index 1 refers to gini values
    max_value = gini_values.max()
    index = np.where(gini_values == max_value)[0][0]

    val = k.iloc[:, [index]].loc[0]
    best_value = val.values[0]
    best_feature = val.index[0]
    return best_feature, best_value


def best_split_value(series, y, split_type):
    values = series.unique()
    inf_gain_values = []
    feature_values = []

    for val in values:
        if split_type == '<=':
            mask = series < val
    else:
        mask = series >= val

    i_g = information_gain(y, mask, func=entropy)
    inf_gain_values.append(i_g)
    feature_values.append(val)

    max_inf_gain = np.max(inf_gain_values)
    index = inf_gain_values.index(max_inf_gain)
    best_feature_value = feature_values[index]

    return best_feature_value, max_inf_gain


def information_gain(y, mask, func=entropy):
    """Information gain is used for determining the best
   features/attributes that render maximum information about a class.

   :param y: (list) target values
   :param mask:(list of booleans) used for splitting the data
   :param func: (function) function for measurements of impurity/uncertainty
   :return: (float) Information Gain
   """
    if isinstance(y, pd.Series):
        y_right = y[mask]
        y_left = y[-mask]
        size = len(y)
        w_right, w_left = len(y_right) / size, len(y_left) / size
        return entropy(y) - (w_right * entropy(y_right) + w_left * w_left)

    else:
        raise TypeError("y must be a Pandas Series")


def best_split(x, y):
    k = x.apply(best_split_value, y=y.squeeze(), split_type=split_type)

    gini_values = k.iloc[1]  # index 1 refers to gini values
    max_value = gini_values.max()
    index = np.where(gini_values == max_value)[0][0]

    val = k.iloc[:, [index]].loc[0]
    best_value = val.values[0]
    best_feature = val.index[0]
    return best_feature, best_value


def split_data(x, y, question):
    feature, value = list(question.split(left_split_type))
    if left_split_type == '<':
        mask = x[feature] < float(value)
    else:
        mask = x[feature] >= value

    x_left = x[mask]
    x_right = x[-mask]
    y_left = y[mask]
    y_right = y[-mask]

    return x_left, y_left, x_right, y_right
