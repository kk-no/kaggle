import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
from datetime import datetime
import random

# titanic_demo
# demo file

# Data Column
# PassengerId Survived Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked

def missing_table(df: pd.DataFrame):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    missing_table = pd.concat([null_val, percent], axis=1)
    missing_table_len_columns = missing_table.rename(
        columns = {0: "欠損数", 1: "%"}
    )
    return missing_table_len_columns

# log
# {'max_depth': 15, 'min_samples_split': 30, 'n_estimators': 25, 'random_state': 0} 77.0%
# {'max_depth': 12, 'min_samples_split': 19, 'n_estimators': 19, 'random_state': 0} 76.0%
# {'max_depth': 10, 'min_samples_split': 25, 'n_estimators': 15, 'random_state': 0} 78.9%
def grid_search_forest(train_data, test_data):
    params = {
        "n_estimators": [5, 10, 15, 20, 25, 30],
        "random_state": [0],
        "min_samples_split": [5, 10, 15, 20, 25, 30],
        "max_depth": [5, 10, 15, 20, 25, 30]
    }

    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params)
    grid_search.fit(train_data, test_data)
    
    print(grid_search.best_score_)
    print(grid_search.best_params_)

# 読込
train: pd.DataFrame = pd.read_csv("csv/train.csv")
test: pd.DataFrame = pd.read_csv("csv/test.csv")

# print("############################################################")
# print(train.head(5))
# print(train.info())
# print("############################################################")
# print(test.head(5))
# print(test.info())
# print("############################################################")

# iterator
train_ = [train]
test_ = [test]

# 前処理
train = train.replace("male", 0).replace("female", 1).replace("S", 0).replace("C", 1).replace("Q", 2)
test = test.replace("male", 0).replace("female", 1).replace("S", 0).replace("C", 1).replace("Q", 2)

# 欠損値埋め
# inplace=Trueでデータ元を更新
train["Age"].fillna(train.Age.mean(), inplace=True)
train["Embarked"].fillna(train.Embarked.mean(), inplace=True)

test["Age"].fillna(test.Age.mean(), inplace=True)
test["Fare"].fillna(test.Fare.mean(), inplace=True)

# 変数追加
train["Family"] = train["SibSp"] + train["Parch"] + 1
for tr in train_:
    train["IsAlone"] = 0
    train.loc[train["Family"] == 1, "IsAlone"] = 1

test["Family"] = test["SibSp"] + test["Parch"] + 1
for te in test_:
    test["IsAlone"] = 0
    test.loc[test["Family"] == 1, "IsAlone"] = 1

# print(train.head(10))
# print(test.head(10))

# 学習データ(train)
x_train = train[["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Family", "IsAlone"]].values
# 正解データ(train)
y_train = train[["Survived"]].values

# 学習データ(test)
x_test = test[["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Family", "IsAlone"]].values

# grid_search_forest(x_train, y_train.ravel())
# exit()

# 決定木作成
forest = RandomForestClassifier(max_depth=10, min_samples_split=25, n_estimators=15, random_state=0)
forest.fit(x_train, y_train.ravel())
# 予測
predicted_label = forest.predict(x_test)

# train["Age"] = train["Age"].fillna(train["Age"].median())
# train.Fare[train.Fare == 0] = train.Fare.median()
# train["Embarked"] = train["Embarked"].fillna("S")

# train.Sex[train.Sex == "male"] = 0
# train.Sex[train.Sex == "female"] = 1
# train.Embarked[train.Embarked == "S"] = 0
# train.Embarked[train.Embarked == "C"] = 1
# train.Embarked[train.Embarked == "Q"] = 2

# test["Age"] = test["Age"].fillna(test["Age"].median())
# test["Fare"][test["Fare"] == 0] = test["Fare"].median()
# test["Embarked"] = test["Embarked"].fillna("S")

# test["Sex"][test["Sex"] == "male"] = 0
# test["Sex"][test["Sex"] == "female"] = 1
# test["Embarked"][test["Embarked"] == "S"] = 0
# test["Embarked"][test["Embarked"] == "C"] = 1
# test["Embarked"][test["Embarked"] == "Q"] = 2
# test.Fare[152] = test.Fare.median()

# target = train["Survived"].values
# train_features = train[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values
# test_features = test[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]].values

# max_depth rate
# 7 75%
# 9 76%
# 10 76%
# 11 70%
# 13 71%
# 21 70%

# decision_tree = tree.DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=0)
# decision_tree = decision_tree.fit(train_features, target)
# predicted_label  = decision_tree.predict(test_features)

# for i in range(2, 15):
    # depth = 10
    # depth = random.randint(1, 10)
    # min_split = i
    # min_split = random.randint(2, 15)
    # decision_tree = tree.DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, random_state=1)
    # decision_tree = decision_tree.fit(train_features, target)
    # predicted_label  = decision_tree.predict(train_features)
    # score = accuracy_score(target, predicted_label)
    # print("{0} depth={1} min_split={2}: {3}".format(i, depth, min_split, score))

passengerId = np.array(test["PassengerId"]).astype(int)
solution: pd.DataFrame = pd.DataFrame(predicted_label, passengerId, columns = ["Survived"])
solution.to_csv("{0:%Y%m%d%H%M%S}.csv".format(datetime.now()), index_label=["PassengerId"])