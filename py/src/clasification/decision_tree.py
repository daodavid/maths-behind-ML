import numpy as np
import pandas as pd
import collections

left_split_type = '<'
right_split_type = '>='


class DecisionTree:

    def __init__(self, max_depth, min_ginni_index=0):
        self.max_depth = max_depth
        self.root = None
        self.min_ginni_index = min_ginni_index

    def train(self, x, y):
        verify(x, y)
        self.root = self.Node(question=None, depth=0)
        self._append_nodes(self.root, x, y)
        return self.root

    def predict(self, x):
        predict = x.apply(self._calc, child=self.root, axis=1)
        return predict

    @staticmethod
    def accuracy(y, predict):
        k = y.T.to_numpy()
        c = predict.to_numpy()
        d = (k == c).astype(int).sum()
        v = k.shape[1]
        result = d / v
        return result

    def _calc(self, row, child):
        question = child.question
        if question:
            feature, value = list(question.split(left_split_type))
            is_true = row.loc[feature] < float(value)
            if is_true:
                return self._calc(row, child.left_child)
            else:
                return self._calc(row, child.right_child)
        else:
            return child.response

    def _stop(self, node):
        pass

    def _append_nodes(self, parent, x, y):
        if (not self.max_depth <= parent.depth and len(y) > 2) and not entropy(y) <= self.min_ginni_index:
            feature, value, max_ig = best_split(x, y)
            parent.question = str(feature) + left_split_type + str(value)
            x_left, y_left, x_right, y_right = split_data(x, y, parent.question)
            parent.left_child = self.Node(None, self._response(y_left), parent.depth + 1)
            parent.right_child = self.Node(None, self._response(y_right), parent.depth + 1)
            self._append_nodes(parent.left_child, x_left, y_left)
            self._append_nodes(parent.right_child, x_right, y_right)
        else:
            return

    @staticmethod
    def _response(y):
        number_true = y.sum()[0]
        number_false = y.shape[0] - number_true
        return (number_true > number_false).astype('int')

    class Node:
        def __init__(self, question=None, response=0, depth=0):
            self.left_child = None
            self.right_child = None
            self.question = question
            self.response = response
            self.depth = depth


def entropy(y):
    """ Entropy measures the degree of uncertainty,
     impurity or disorder of a random variable

    :param target: (list) target values
    :return: (float)  entropy of the array
    """

    counter = collections.Counter(y)
    p = y.value_counts().to_numpy() / len(y)
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
    return best_feature, best_value, max_value


def best_split_value(series, y, split_type):
    values = series.unique()
    inf_gain_values = []
    feature_values = []

    for val in values:
        if split_type == '<':
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
        information_gain = entropy(y) - (w_right * entropy(y_right) + w_left * entropy(y_left))
        return information_gain

    else:
        raise TypeError("y must be a Pandas Series")


def best_split(x, y):
    k = x.apply(best_split_value, y=y.squeeze(), split_type=left_split_type)

    gini_values = k.iloc[1]  # index 1 refers to gini values
    max_value = gini_values.max()
    index = np.where(gini_values == max_value)[0][0]
    val = None
    try:
        val = k.iloc[:, [index]].loc[0]
    except Exception:
        print('Caught Value Exception : ' + val)

    best_value = val.values[0]
    best_feature = val.index[0]
    return best_feature, best_value, max_value


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


data = pd.read_csv("../../../resources/data/500_Person_Gender_Height_Weight_Index.csv")

#data.drop('Gender', axis=1, inplace=True)
print(data)
data['obese'] = (data.Index > 4).astype('int')
print(data)
data.drop('Index', axis=1, inplace=True)
x = data.drop(['obese'], axis=1)
x = pd.get_dummies(x, dummy_na=True)
y = data[['obese']]
print(data)
tree = DecisionTree(max_depth=10)

tree.train(x, y)
y_predict = tree.predict(x)
acc = tree.accuracy(y, y_predict)
print(acc)
