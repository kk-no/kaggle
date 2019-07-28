import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# titanic
# RandomForest 

# Data Column
# PassengerId Survived Pclass Name Sex Age SibSp Parch Ticket Fare Cabin Embarked

# 20190724 71%
# 20190725 76%
# 20190727 78.9%

# 読込
train: pd.DataFrame = pd.read_csv("csv/train.csv")
test: pd.DataFrame = pd.read_csv("csv/test.csv")

# イテレータ
train_ = [train]
test_ = [test]

# 前処理
train = train.replace("male", 0).replace("female", 1).replace("S", 0).replace("C", 1).replace("Q", 2)
test = test.replace("male", 0).replace("female", 1).replace("S", 0).replace("C", 1).replace("Q", 2)

# 欠損値埋め
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

# 学習データ(train)
x_train = train[["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Family", "IsAlone"]].values
# 正解データ(train)
y_train = train[["Survived"]].values

# 学習データ(test)
x_test = test[["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Family", "IsAlone"]].values

# 決定木作成
forest = RandomForestClassifier(max_depth=10, min_samples_split=25, n_estimators=15, random_state=0)
forest.fit(x_train, y_train.ravel())
# 予測
predicted_label = forest.predict(x_test)

# 列抽出
passengerId = np.array(test["PassengerId"]).astype(int)
# 整形
solution: pd.DataFrame = pd.DataFrame(predicted_label, passengerId, columns = ["Survived"])
# 出力
solution.to_csv("{0:%Y%m%d%H%M%S}.csv".format(datetime.now()), index_label=["PassengerId"])