from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, SGD
import os

# dogimage demo

# ファイル名取得    
images = os.listdir("data/all-dogs/all-dogs")
breeds = os.listdir("data/Annotation/Annotation")

# model summary
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_1 (Dense)              (None, 2048)              20482048
# _________________________________________________________________
# reshape_1 (Reshape)          (None, 8, 8, 32)          0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 8, 8, 128)         36992
# _________________________________________________________________
# up_sampling2d_1 (UpSampling2 (None, 16, 16, 128)       0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 16, 16, 64)        73792
# _________________________________________________________________
# up_sampling2d_2 (UpSampling2 (None, 32, 32, 64)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 32, 32, 32)        18464
# _________________________________________________________________
# up_sampling2d_3 (UpSampling2 (None, 64, 64, 32)        0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 64, 64, 3)         867
# =================================================================
# Total params: 20,612,163
# Trainable params: 20,612,163
# Non-trainable params: 0
# _________________________________________________________________
# None

direct_input = ((10000, ))
model = Sequential()
# 全結合層
model.add(Dense(2048, activation="elu", input_shape=direct_input))
# 出力整形
model.add(Reshape((8, 8, 32)))
# 畳みこみ層(2次元)
model.add(Conv2D(128, (3, 3), activation="elu", padding="same"))
# アップサンプリング
model.add(UpSampling2D(2))
# 畳みこみ層(2次元)
model.add(Conv2D(64, (3, 3), activation="elu", padding="same"))
# アップサンプリング
model.add(UpSampling2D(2))
# 畳みこみ層(2次元)
model.add(Conv2D(32, (3, 3), activation="elu", padding="same"))
# アップサンプリング
model.add(UpSampling2D(2))
# 畳みこみ層(2次元)
model.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same"))

# コンパイル
model.compile(optimizer="adam", loss="binary_crossentropy")
# プロット
print(model.summary())