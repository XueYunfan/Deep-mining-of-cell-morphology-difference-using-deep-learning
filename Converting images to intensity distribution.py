import numpy as np
import tensorflow as tf
import os
import pandas as pd

IMAGE_PATH = 'XXX'
SAVE_PATH = 'XXX'

image_name = os.listdir(IMAGE_PATH)

intensities = np.zeros((2700,255))
image_num=0

for name in image_name:
	filename = IMAGE_PATH+name
	image_contents = tf.io.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_contents)
	image_processed = tf.image.random_brightness(image_decoded, max_delta=0.1)
	image_processed = tf.image.random_contrast(image_processed, 0.7, 1.5)
	image_converted = tf.cast(image_processed, tf.float32)
	image = image_converted.numpy()
	for intensity in range(0, 256):
		intensities[n,intensity] = np.sum(image[:,:,1]==intensity)
	image_num += 1

intensities = pd.DataFrame(data=intensities)
intensities.to_csv('XXX.csv',encoding='gbk')
