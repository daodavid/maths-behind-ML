import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("../../../resources/data/500_Person_Gender_Height_Weight_Index.csv")
print(data.head())

data['obese'] = (data.Index > 4).astype('int')
data.drop('Index', axis=1, inplace=True)
print(data)


def gini_impurity(y):
    """
    Given a Pandas Series, it calculates the Gini Impurity.
    y: variable with which calculate Gini Impurity.
    """
    if isinstance(y, pd.Series):
        print(y.value_counts())
        p = y.value_counts() / y.shape[0]
        gini = 1 - np.sum(p ** 2)
        return gini

    else:
        raise Exception('object must be a Pandas Series')


gini_impurity(data.Gender)


def entropy(y):
    """
    Given a Pandas Series, it calculates the Entropy
    y: variable with wich calculate Entropy
    """
    if isinstance(y, pd.Series):
        p = y.value_counts() / y.shape[0]
        entropy = np.sum(-p * np.log2(p) + 1e-9)
        return entropy
    else:
        raise Exception("Object y must to be a Pandas Series")


entropy(data['obese'][data['Weight'] >= 140])

def variance(y):

    """
    Function to help calculate the variance avoiding nan.
    y: variable to calculate variance to. It should be a Pandas Series.
    """

    if len(y) == 1:
        return 0
    else:
        return y.var()


def information_gain(y, mask, func=entropy):
    """
    It returns the Information Gain of a variable given a loss function.
    y: target variable.
    mask: split choice.
    func: function to be used to calculate Information Gain in case os classification.
    """

    a = sum(mask)
    b = mask.shape[0] - a

    if a == 0 or b == 0:
        ig = 0

    else:
        if   False :# y.dtypes != 'O':
            ig = variance(y) - (a / (a + b) * variance(y[mask])) - (b / (a + b) * variance(y[-mask]))
        else:
            ig = func(y) - a / (a + b) * func(y[mask]) - b / (a + b) * func(y[-mask])

    return ig


def max_information_gain_split(x, y, func=entropy):
    x, y = data['Weight'], data['obese']

    split_value = []
    ig = []

    numeric_variable = True if x.dtypes != 'O' else False  # check the type of column

    # Create options according to variable type
    if numeric_variable:
        options = x.sort_values().unique()[1:]
    else:
        pass

    for val in options:
        # print(x)
        mask = x < val if numeric_variable else x.isin(val)
        val_ig = information_gain(y, mask, func=entropy)
        print(val_ig)

        ig.append(val_ig)
        split_value.append(val)

        if len(ig) == 0:
            return None, None, None, False

        # get results with highest IG
    best_ig = min(ig)
    best_ig_index = ig.index(best_ig)
    best_split_number = split_value[best_ig_index]
    return best_ig, best_split_number, numeric_variable, True


weight_ig, weight_slpit, _, _ = max_information_gain_split(data['Weight'], data['obese'], )
print(weight_slpit)
