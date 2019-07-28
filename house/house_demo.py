import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


# house
# 

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

# pandas option
pd.options.display.max_rows = 20
pd.options.display.max_columns = 100

# 読込
train: pd.DataFrame = pd.read_csv("csv/train.csv")
test: pd.DataFrame = pd.read_csv("csv/test.csv")

# 学習データとテストデータのマージ
train['WhatIsData'] = 'Train'
test['WhatIsData'] = 'Test'
test['SalePrice'] = 9999999999
alldata = pd.concat([train,test], sort=False, axis=0).reset_index(drop=True)

# print(train.info())

# 欠損状況
# print(train.isnull().sum()[train.isnull().sum() > 0])
# print(test.isnull().sum()[test.isnull().sum() > 0])

no_col_list = alldata.isnull().sum()[alldata.isnull().sum() > 0].index.tolist()
# print(alldata[no_col_list].dtypes.sort_values())

# float型欠損値
no_float_list = alldata[no_col_list].dtypes[alldata[no_col_list].dtypes == "float64"].index.tolist()
# object型欠損値
no_obj_list = alldata[no_col_list].dtypes[alldata[no_col_list].dtypes == "object"].index.tolist()

# 欠損値埋め
for no_float_col in no_float_list:
    # 数値変数
    alldata.loc[alldata[no_float_col].isnull(), no_float_col] = 0.0
for no_obj_col in no_obj_list:
    # カテゴリカル変数
    alldata.loc[alldata[no_obj_col].isnull(), no_obj_col] = "NA"

# 確認
# print(alldata.isnull().sum()[alldata.isnull().sum() > 0])
# exit()

# 特徴量リスト化
cat_cols = alldata.dtypes[alldata.dtypes == "object"].index.tolist()
num_cols = alldata.dtypes[alldata.dtypes != "object"].index.tolist()

# 分割リスト
other_cols = [
    "Id",
    "WhatIsData"
]

# 不要列削除
cat_cols.remove("WhatIsData")
num_cols.remove("Id")

# カテゴリカル変数のダミー化
dummy_cat = pd.get_dummies(alldata[cat_cols])

# データ統合
all_data = pd.concat([alldata[other_cols], alldata[num_cols], dummy_cat], sort=False, axis=1)
