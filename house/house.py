# house
# RandomForestRegressor

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