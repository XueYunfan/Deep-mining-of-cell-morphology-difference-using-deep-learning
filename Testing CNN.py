import os
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix

#load test set
TEST_PATH = 'XXX'

test_names = os.listdir(TEST_PATH)
test_file = []
test_labels = []

for name in test_names:
	test_file.append(TEST_PATH+name)
	if name[0] == '2':
		test_labels.append(0)
	elif name[0] == '4':
		test_labels.append(1)
	elif name[0] == '6':
		test_labels.append(2)
		
#define parse function
def parse_function_test(filename, label):
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_processed = tf.image.random_brightness(image_processed, max_delta=0.1)
	image_processed = tf.image.random_contrast(image_processed, 0.7, 1.5)
	image_converted = tf.cast(image_processed, tf.float32)
	image_scaled = tf.divide(image_converted, 255.0)
	return image_scaled, label

#generate batch
test_filenames = tf.constant(test_file)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, 3, dtype='int32')
test_labels_onehot = tf.constant(test_labels_onehot)
test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels_onehot))
test_dataset = test_dataset.map(parse_function_test)
test_dataset = test_dataset.batch(4)

#load model and predict
model_path = 'XXX.hdf5'
model = load_model(model_path)
history = model.evaluate(test_dataset)

predictions = model.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)

for results in zip(test_file, predictions):
	print(results)

#draw confuion matrix
cm = confusion_matrix(test_labels, predictions)
classes = ['1:20','1:40','1:60']

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_yticks(range(3))
	ax.set_yticklabels(classes)
	ax.set_xticks(range(3))
	ax.set_xticklabels(classes)
	plt.tick_params(labelsize=10)
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title, fontsize=13)
	plt.colorbar()
	
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
		horizontalalignment="center",
		color="white" if cm[i, j] > thresh else "black",
		fontsize = 10)

	plt.tight_layout()
	plt.ylabel('True Label',fontsize=13)
	plt.xlabel('Predicted Label',fontsize=13)
	plt.tight_layout()
	plt.savefig('XXX', dpi=300)

plot_confusion_matrix(cm,classes)
