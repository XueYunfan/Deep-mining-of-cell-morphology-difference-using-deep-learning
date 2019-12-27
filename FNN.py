import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import training_hist_vis

#load data
data = np.loadtxt(
	open("Image intensity distribution.csv","rb"),delimiter=",",skiprows=0,dtype=float)
intensity = data[:,:256]
target = data[:,256]
target = tf.keras.utils.to_categorical(peptide_mic, 3).astype(dtype=np.int32)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
	intensity, target, test_size=0.2, random_state=1)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#define model and train
FNN_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(256, activation='relu', use_bias=True),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(512, activation='relu', use_bias=True),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(512, activation='relu', use_bias=True),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(256, activation='relu', use_bias=True),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(3, activation='softmax')
])

FNN_model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

model_save_path = 'XXX'
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path, 
	monitor='val_accuracy', verbose=0, save_best_only=True, 
	save_weights_only=False, mode='max', period=1)

callback = [checkpoint]

hist = FNN_model.fit(X_train_scaled, y_train, validation_split=0.15,
	epochs=300, batch_size=4, callbacks=callback)
#====================================================
training_vis(hist)
best_model = load_model(model_save_path)
best_model.evaluate(X_test_scaled, y_test)
