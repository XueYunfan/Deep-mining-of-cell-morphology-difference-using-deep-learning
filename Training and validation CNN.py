import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
import training_hist_vis

#load dataset
TRAINFILE_PATH = 'XXX'
VALIDATIONFILE_PATH = 'XXX'

train_names = os.listdir(TRAINFILE_PATH)
validation_names = os.listdir(VALIDATIONFILE_PATH)

train_file = []
validation_file = []

train_labels = []
validation_labels = []

for name in train_names:
	train_file.append(TRAINFILE_PATH+name)
	if name[0] == '2':
		train_labels.append(0)
	elif name[0] == '4':
		train_labels.append(1)
	elif name[0] == '6':
		train_labels.append(2)

for name in validation_names:
	validation_file.append(VALIDATIONFILE_PATH+name)
	if name[0] == '2':
		validation_labels.append(0)
	elif name[0] == '4':
		validation_labels.append(1)
	elif name[0] == '6':
		validation_labels.append(2)
		
#define parse function
def parse_function_train(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_processed = tf.image.random_flip_left_right(image_decoded)
	image_processed = tf.image.random_flip_up_down(image_processed)
	image_processed = tf.image.random_brightness(image_processed, max_delta=0.1)
	image_processed = tf.image.random_contrast(image_processed, 0.7, 1.5)
	image_converted = tf.cast(image_processed, tf.float32)
	image_scaled = tf.divide(image_converted, 255.0)
	return image_scaled, label

def parse_function_test(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_converted = tf.cast(image_decoded, tf.float32)
	image_scaled = tf.divide(image_converted, 255.0)
	return image_scaled, label

#generate batch
train_filenames = tf.constant(train_file)
train_labels = tf.keras.utils.to_categorical(train_labels, 3, dtype='int32')
train_labels = tf.constant(train_labels)
validation_filenames = tf.constant(validation_file)
validation_labels = tf.keras.utils.to_categorical(validation_labels, 3, dtype='int32')
validation_labels = tf.constant(validation_labels)

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.shuffle(len(train_file))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_filenames, validation_labels))

train_dataset = train_dataset.map(parse_function_train)
validation_dataset = validation_dataset.map(parse_function_test)

train_dataset = train_dataset.batch(4).repeat()
validation_dataset = validation_dataset.batch(4)

#define model
base_model = tf.keras.applications.ResNet50V2(
	weights='imagenet', include_top=False, input_shape=(536,536,3), pooling='avg')

output1 = base_model.output
predictions = tf.keras.layers.Dense(3, activation='softmax')(output1)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers:
   layer.trainable = True

sgd = tf.keras.optimizers.SGD(lr=0, momentum=0.9, decay=0, nesterov=False)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
		
model.summary()

#define learning rate decay and train model
def lrate_decay(epoch):
    init_lrate = 0.001
    drop = 0.5
    lrate = init_lrate * pow(drop, (epoch//10))
    print('lrate = '+str(lrate))
    return lrate

lrate = LearningRateScheduler(lrate_decay)

savepath = 'XXX.hdf5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(savepath, 
	monitor='val_accuracy', verbose=0, save_best_only=True, 
	save_weights_only=False, mode='max', period=1)

callback = [lrate, checkpoint]

hist = model.fit(train_dataset, validation_data=validation_dataset, 
		epochs=100, steps_per_epoch=473, callbacks=callback)

model.save('XXX.hdf5')
training_vis(hist)
