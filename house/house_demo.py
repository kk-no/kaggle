import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns

# house demo file
# RandomForestRegressor

# 1460 record 

# DataColumn
# Id MSSubClass MSZoning LotFrontage LotArea 
# Street Alley LotShape LandContour Utilities 
# LotConfig LandSlope Neighborhood Condition1 Condition2 
# BldgType HouseStyle OverallQual OverallCond YearBuilt 
# YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd 
# MasVnrType MasVnrArea ExterQual ExterCond Foundation 
# BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinSF1 
# BsmtFinType2 BsmtFinSF2 BsmtUnfSF TotalBsmtSF Heating 
# HeatingQC CentralAir Electrical 1stFlrSF 2ndFlrSF 
# LowQualFinSF GrLivArea BsmtFullBath BsmtHalfBath FullBath 
# HalfBath BedroomAbvGr KitchenAbvGr KitchenQual TotRmsAbvGrd 
# Functional Fireplaces FireplaceQu GarageType GarageYrBlt 
# GarageFinish GarageCars GarageArea GarageQual GarageCond 
# PavedDrive WoodDeckSF OpenPorchSF EnclosedPorch 3SsnPorch 
# ScreenPorch PoolArea PoolQC Fence MiscFeature 
# MiscVal MoSold YrSold SaleType SaleCondition 
# SalePrice

# grid search
def grid_search_forest(train_data: pd.DataFrame, test_data: pd.DataFrame):
    params = {
        "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "random_state": [0],
        "min_samples_split": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "max_depth": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    }
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid=params)
    grid_search.fit(train_data, test_data)

    print(grid_search.best_score_)
    print(grid_search.best_params_)

# correlation check
def check_correlation(train_data: pd.DataFrame):
    # 基準とする相関係数
    border = 0.5

    # 相関係数の算出
    co = train_data.corr()
    # cl = train_data.columns
    # heat = sns.heatmap(
    #     co,
    #     cbar=True,
    #     square=True,
    #     fmt=".2f",
    #     annot_kws={'size': 15},
    #     yticklabels=cl,
    #     xticklabels=cl,
    #     cmap='Accent'
    # )
    # plt.show()
    correlation_co: pd.DataFrame = co.loc[co["SalePrice"] > border]
    print(correlation_co.index)
    exit()

# pandas option
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100

# 読込
train: pd.DataFrame = pd.read_csv("csv/train.csv")
test: pd.DataFrame = pd.read_csv("csv/test.csv")

# 学習データ欠損列
train_no_col_list = train.isnull().sum()[train.isnull().sum() > 0].index.tolist()
# テストデータ欠損列
test_no_col_list = test.isnull().sum()[test.isnull().sum() > 0].index.tolist()
# print(train[train_no_col_list].dtypes.sort_values())
# print(test[test_no_col_list].dtypes.sort_values())

# 学習データ数値変数欠損列
train_no_float_cols = train[train_no_col_list].dtypes[train[train_no_col_list].dtypes == "float64"].index.tolist()
# 学習データカテゴリカル変数欠損列
train_no_obj_cols = train[train_no_col_list].dtypes[train[train_no_col_list].dtypes == "object"].index.tolist()

# テストデータ数値変数欠損列
test_no_float_cols = test[test_no_col_list].dtypes[test[test_no_col_list].dtypes == "float64"].index.tolist()
# テストデータカテゴリカル変数欠損列
test_no_obj_cols = test[test_no_col_list].dtypes[test[test_no_col_list].dtypes == "object"].index.tolist()

# print(train_no_float_cols)
# print(train_no_obj_cols)
# print(test_no_float_cols)
# print(test_no_obj_cols)

# 学習データ欠損値埋め
for train_no_float_col in train_no_float_cols:
    # 数値変数
    train.loc[train[train_no_float_col].isnull(), train_no_float_col] = 0.0

for train_no_obj_col in train_no_obj_cols:
    # カテゴリカル変数
    train.loc[train[train_no_obj_col].isnull(), train_no_obj_col] = "NA"

# 確認
# print(train.isnull().sum()[train.isnull().sum() > 0])

# テストデータ欠損値埋め
for test_no_float_col in test_no_float_cols:
    # 数値変数
    test.loc[test[test_no_float_col].isnull(), test_no_float_col] = 0.0

for test_no_obj_col in test_no_obj_cols:
    # カテゴリカル変数
    test.loc[test[test_no_obj_col].isnull(), test_no_obj_col] = "NA"

# 確認
# print(test.isnull().sum()[test.isnull().sum() > 0])

# カテゴリカル変数のダミー化
for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == "object":
        lbl = LabelEncoder()
        # ラベル抽出
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        # 学習データ変換
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        # テストデータ置換
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

# print(train.info())
# print(test.info())

# 相関チェック
check_correlation(train)

# 学習データ(train)
x_train = train.drop(["Id", "SalePrice"], axis=1)
# 正解データ(train)
y_train = train["SalePrice"]

# 予測対象
x_test = test.drop(["Id"], axis=1)

# GridSearch
# grid_search_forest(x_train, y_train)
# exit()

# 決定木作成
forest = RandomForestRegressor(max_depth=20, min_samples_split=5, n_estimators=100, random_state=0)
forest.fit(x_train, y_train)

# 予測
predicted_label = forest.predict(x_test)

# print(predicted_label)

# ID列抽出
Id = np.array(test["Id"]).astype(int)
# データフレーム作成
solution = pd.DataFrame(predicted_label, Id, columns = ["SalePrice"])
# 出力
solution.to_csv("{0:%Y%m%d%H%M%S}.csv".format(datetime.now()), index_label=["Id"])