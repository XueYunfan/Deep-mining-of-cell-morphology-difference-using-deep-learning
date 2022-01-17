# -*- coding: UTF-8
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import numpy as np
import cv2
tf.compat.v1.disable_eager_execution()

def grad_cam(model, img_name, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    image_contents = tf.io.read_file(img_name)
    image_decoded = tf.image.decode_jpeg(image_contents)
    image_converted = tf.cast(image_decoded, tf.float32)
    image_scaled = tf.divide(image_converted, 255.0)
    x = tf.reshape(image_scaled,(1,536,536,3))
    
    # 取得目标分类的CNN输出loss
    class_output = model.output[:, category_index]

    # 取得想要算出梯度的层的输出
    convolution_output = model.get_layer(layer_name).output
    # 利用gradients函数，算出梯度公式
    grads = K.gradients(class_output, convolution_output)[0]
    # 定义计算函数
    gradient_function = K.function([model.input], [convolution_output, grads])
    # 根据实际的输入图像得出梯度张量
    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    # 取得所有梯度的平均值
    weights = np.mean(grads_val, axis=(0, 1))

    # 把所有平面的平均梯度乘到最后卷积层上，得到一个影响输出的梯度权重图
    cam = np.dot(output, weights)

    # 把梯度权重图RGB化
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image_rgb = cv2.imread(img_name)
    b,g,r = cv2.split(image_rgb)
    image_rgb = cv2.merge([g,g,g])

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam)*0.25 + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), np.uint8(heatmap*255)


model = tf.keras.models.load_model('EC/best model for EC-FITC on PDMS ResNet50V2.hdf5')
model.summary()
PATH = 'CAMtests/EC/'
names = os.listdir(PATH)

for name in names:
	if name[:2]=='20':
		cate=0
	elif name[:2]=='40':
		cate=1
	else:
		cate=2
	a,b = grad_cam(model,PATH+name,cate,'conv5_block3_out')
	cv2.imwrite('CAMtests/EC_CAM/{}'.format(name),a)
