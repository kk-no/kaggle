import pandas as pd
import numpy as np
from sklearn import tree

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
train_features = train[["Pclass", "Sex", "Age", "Fare"]].values
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(train_features, target)
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

prediction  = decision_tree.predict(test_features)
# print(prediction.shape)

PassengerId = np.array(test["PassengerId"]).astype(int)
solution: pd.DataFrame = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
solution.to_csv("tree.csv", index_label=["PassengerId"])