import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# mercari demo file
# LightGBM

# pandas option
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# 読込
train: pd.DataFrame = pd.read_csv("data/train.tsv", delimiter='\t')
test: pd.DataFrame = pd.read_csv("data/test_stg2.tsv", delimiter='\t')


# 学習データ欠損列
train_no_col_list = train.isnull().sum()[train.isnull().sum() > 0].index.tolist()
# テストデータ欠損列
test_no_col_list = test.isnull().sum()[test.isnull().sum() > 0].index.tolist()

# object only
# train['category_name', 'brand_name', 'item_description']
# test['category_name', 'brand_name', 'item_description']

# 学習データ欠損値埋め
for train_no_col in train_no_col_list:
    # カテゴリカル変数
    # train.loc[train[train_no_col].isnull(), train_no_col] = "NaN"
    train[train_no_col] = np.where(train[train_no_col].isnull(), 0, 1)

# テストデータ欠損値埋め
for test_no_col in test_no_col_list:
    # カテゴリカル変数
    # test.loc[test[test_no_col].isnull(), test_no_col] = "NaN"
    test[test_no_col] = np.where(test[test_no_col].isnull(), 0, 1)

# カテゴリカル変数のダミー化
# for i in ["category_name", "brand_name"]:
#     lbl = LabelEncoder()
#     lbl.fit(list(train[i].values) + list(test[i].values))
#     train[i] = lbl.transform(train[i])
#     test[i] = lbl.transform(test[i])

# 学習データ(train)
x_train = train.drop(["train_id", "price", "name"], axis=1)
# 正解データ(train)
y_train = train["price"]

# 予測対象
x_test = test.drop(["test_id", "name"], axis=1)

# モデル作成
model = lgb.LGBMRegressor()
model.fit(x_train, y_train)

# 予測
predicted_label = model.predict(x_test)

# ID抽出
ID = np.array(test["test_id"]).astype(int)
# データフレーム作成
solution = pd.DataFrame(predicted_label, ID, columns = ["price"])
# 出力
solution.to_csv("{0:%Y%m%d%H%M%S}.csv".format(datetime.now()), index_label=["test_id"])
