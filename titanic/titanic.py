import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from datetime import datetime

# 20190723 71%
# 20190724 76%

def missing_table(df: pd.DataFrame):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    missing_table = pd.concat([null_val, percent], axis=1)
    missing_table_len_columns = missing_table.rename(
        columns = {0: "欠損数", 1: "%"}
    )
    return missing_table_len_columns

train: pd.DataFrame = pd.read_csv("csv/train.csv")
test: pd.DataFrame = pd.read_csv("csv/test.csv")

# PassengerId Survived Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked
# print(train.head(10))
# exit()

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# print("##### train #####")
# print(train.shape)
# print(train.head())
# print(train.describe())
# print(missing_table(train))

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"][test["Fare"] == 0] = test["Fare"].median()
test["Embarked"] = test["Embarked"].fillna("S")

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()

# print("##### test #####")
# print(test.shape)
# print(test.head())
# print(test.describe())
# print(missing_table(test))

target = train["Survived"].values
train_features = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values

# 7 75%
# 9 76%
# 10 76%
# 11 70%
# 13 71%
# 21 70%
decision_tree = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=0)
decision_tree = decision_tree.fit(train_features, target)
predicted_label  = decision_tree.predict(test_features)
# for depth in range(1, 36):
#     decision_tree = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=5, random_state=1)
#     decision_tree = decision_tree.fit(train_features, target)
#     predicted_label  = decision_tree.predict(train_features)
#     score = accuracy_score(target, predicted_label)
#     print("depth={0}: {1}".format(depth, score))

PassengerId = np.array(test["PassengerId"]).astype(int)
solution: pd.DataFrame = pd.DataFrame(predicted_label, PassengerId, columns = ["Survived"])
solution.to_csv("{0:%Y%m%d%H%M%S}.csv".format(datetime.now()), index_label=["PassengerId"])