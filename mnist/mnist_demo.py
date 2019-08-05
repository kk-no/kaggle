import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

train: pd.DataFrame = pd.read_csv("data/train.csv")
test: pd.read_csv = pd.read_csv("data/test.csv")

# 学習データ
x_train = train.drop(["label"], axis=1)
# 正解ラベル
y_train = train["label"]

# 予測対象
x_test = test

# メモリ解放
del train
del test

# 学習データ形状変換
x_train = x_train / 255.0
x_train = x_train.values.reshape(-1, 28, 28, 1)

# 予測対象形状変換
x_test = x_test / 255.0
x_test = x_test.values.reshape(-1, 28, 28, 1)

# One-Hotラベルの生成
y_train = to_categorical(y_train, num_classes=10)

# 学習データ、テストデータ、訓練正解ラベル、テスト正解ラベル
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

# CNN model
# model = Sequential()

# model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="Same", activation="relu", input_shape=(28, 28, 1)))
# model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="Same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="Same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation="softmax"))

# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(X_train, Y_train, batch_size=136, epochs=10, validation_data=(X_test, Y_test), verbose=2)

# print(model.summary())
model = load_model("mnist_cnn.h5")

prediction: pd.DataFrame = model.predict(x_test)

prediction = np.argmax(prediction, axis=1)
# prediction = pd.DataFrame(prediction, columns = ["Label"])
prediction = pd.DataFrame({
    "ImageId": list(range(1, len(prediction) + 1)),
    "Label": prediction
})

# 出力
prediction.to_csv("{0:%Y%m%d%H%M%S}.csv".format(datetime.now()), index=False)

# model保存
model.save("mnist_cnn.h5")

# デバッグ
# print(prediction[0])
# plt.imshow(test[0][:, :, 0])
# plt.show()