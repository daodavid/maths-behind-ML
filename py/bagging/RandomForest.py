import category_encoders as ce
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from sklearn.model_selection import train_test_split




from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../../resources/data/car_evaluation.csv")
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# encode categorical variables with ordinal encoding

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

tree = DecisionTreeClassifier()

# Train Decision Tree Classifer
tree = tree.fit( X_train.sample(n=300),y_train)
#Predict the response for test dataset
y_pred = tree.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",accuracy_score(y_test, y_pred))