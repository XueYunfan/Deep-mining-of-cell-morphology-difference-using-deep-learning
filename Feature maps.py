import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
import matplotlib.pyplot as plt

model_path = 'XXX.hdf5'
model = load_model(model_path)
model.summary()

file_path = 'XXX'
img_names = os.listdir(file_path)
img_path = []
images = []

for img_name in img_names:
	img_path.append(img_path+img_name)

for image in img_path:
	img = tf.io.read_file(image)
	img = tf.image.decode_jpeg(img)
	img = img.numpy()
	img = img.reshape(1, 536, 536, 3)
	images.append(img)

model_medium1 = Model(inputs=model.input, outputs=model.get_layer('conv2_block3_1_relu').output)
model_medium2 = Model(inputs=model.input, outputs=model.get_layer('conv3_block4_1_relu').output)
model_medium3 = Model(inputs=model.input, outputs=model.get_layer('conv4_block6_1_relu').output)
model_medium4 = Model(inputs=model.input, outputs=model.get_layer('post_relu').output)

models=[]
models.append(model_medium1)
models.append(model_medium2)
models.append(model_medium3)
models.append(model_medium4)

x=1
for img in images:
	y=1
	for model in models:
		
		medium_output = model.predict(img)
		batch, h, w, channal = medium_output.shape
		feature_map = np.zeros((h,h))
		
		for n in range(channal):
			feature_map += medium_output[0,:,:,n]
			
		plt.imshow(a)
		plt.axis('off')
		fig = plt.gcf()
		fig.savefig('XXX/{}-{}.jpg'.format(str(x),str(y)), dpi=300, bbox_inches = 'tight')
		y+=1
	x+=1
