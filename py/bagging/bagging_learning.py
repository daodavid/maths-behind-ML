import numpy as np
import pandas as pd
import collections

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

print("hello")



df_train = pd.read_csv("../../resources/data/titanic/train.csv")
df_test = pd.read_csv("../../resources/data/titanic/test.csv")

# Store target variable of training data in a safe place
survived_train = df_train[['Survived']]

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# Dealing with missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data = data[['Sex_male', 'Fare', 'Age', 'Pclass', 'SibSp']]


training_data, testing_data = train_test_split(data, test_size=0.2, random_state=25)

tree_1 = DecisionTreeClassifier()

#https://www.kaggle.com/code/ydalat/titanic-a-step-by-step-intro-to-machine-learning



