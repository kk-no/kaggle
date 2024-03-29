import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

# house price
# LightGBM

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

# 読込
train: pd.DataFrame = pd.read_csv("csv/train.csv")
test: pd.DataFrame = pd.read_csv("csv/test.csv")

# 学習データ欠損列
train_no_col_list = train.isnull().sum()[train.isnull().sum() > 0].index.tolist()
# テストデータ欠損列
test_no_col_list = test.isnull().sum()[test.isnull().sum() > 0].index.tolist()

# 学習データ数値変数欠損列
train_no_float_cols = train[train_no_col_list].dtypes[train[train_no_col_list].dtypes == "float64"].index.tolist()
# 学習データカテゴリカル変数欠損列
train_no_obj_cols = train[train_no_col_list].dtypes[train[train_no_col_list].dtypes == "object"].index.tolist()

# テストデータ数値変数欠損列
test_no_float_cols = test[test_no_col_list].dtypes[test[test_no_col_list].dtypes == "float64"].index.tolist()
# テストデータカテゴリカル変数欠損列
test_no_obj_cols = test[test_no_col_list].dtypes[test[test_no_col_list].dtypes == "object"].index.tolist()

# 学習データ欠損値埋め
for train_no_float_col in train_no_float_cols:
    # 数値変数
    train.loc[train[train_no_float_col].isnull(), train_no_float_col] = 0.0

for train_no_obj_col in train_no_obj_cols:
    # カテゴリカル変数
    train.loc[train[train_no_obj_col].isnull(), train_no_obj_col] = "NA"

# テストデータ欠損値埋め
for test_no_float_col in test_no_float_cols:
    # 数値変数
    test.loc[test[test_no_float_col].isnull(), test_no_float_col] = 0.0

for test_no_obj_col in test_no_obj_cols:
    # カテゴリカル変数
    test.loc[test[test_no_obj_col].isnull(), test_no_obj_col] = "NA"

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

# 学習データ(train)
x_train = train.drop(["Id", "SalePrice"], axis=1)

# 正解データ(train)
y_train = train["SalePrice"]

# 予測対象
x_test = test.drop(["Id"], axis=1)

# モデル作成(LightGBM)
model = lgb.LGBMRegressor(random_state=0)
model.fit(x_train, y_train)

# 予測
predicted_label = model.predict(x_test)

# ID列抽出
Id = np.array(test["Id"]).astype(int)
# データフレーム作成
solution = pd.DataFrame(predicted_label, Id, columns = ["SalePrice"])
# 出力
solution.to_csv("{0:%Y%m%d%H%M%S}.csv".format(datetime.now()), index_label=["Id"])