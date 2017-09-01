from keras.layers import Convolution2D, UpSampling2D
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import tflearn

tf.python.control_flow_ops = tf

# Image transformer
datagen = ImageDataGenerator(
		rescale=1.0/255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

# Get images
X = []
for filename in os.listdir('face_images'):
	X.append(img_to_array(load_img('face_images/'+filename)))
X = np.array(X)

# Set up train and test data
split = int(0.9*len(X))
Xtrain = X[:split]
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]

# Set up model
N = 5
model = Sequential()
num_maps1 = [4, 8, 16, 32, 64]
num_maps2 = [8, 16, 32, 64, 128]


def conv_stack(net, channels, strides):
	for s in strides:
		net = tflearn.layers.conv.conv_2d (net, channels, 2, stride=2)
	return net

def dilated_conv_stack(net, channels, dilations):
	for d in dilations:
		net = tflearn.layers.conv.atrous_conv_2d (net, channels, 2, rate=d)

net = tflearn.input_dat(shape=[None, 128, 128, 1])

net = create_conv_layer_stack(net, 64 [1, 2])
net = create_conv_layer_stack(net, 128, [1, 2])
net = create_conv_layer_stack(net, 256, [1, 1, 2])
net = create_conv_layer_stack(net, 512, [1, 1, 1])
net = create_dilated_conv_layer_stack(net, 512, [2, 2, 2])
net = create_dilated_conv_layer_stack(net, 512, [2, 2, 2])
net = create_conv_layer_stack(net, 256, [1, 1, 1])

net = tflearn.layers.conc.conv_2d_transpose (net, 128, 2, [56, 56])
net = create_conv_layer_stack (net, 128, [1,1])

model = tflearn.DNN(net)


batch_size = 10
def image_a_b_gen(batch_size):
	for batch in datagen.flow(Xtrain, batch_size=batch_size):
		if batch == None:
			break		
		lab_batch = rgb2lab(batch)
		X_batch = lab_batch[:,:,:,0]
		Y_batch = lab_batch[:,:,:,1:]
		yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

# Train model
model.fit_generator(
	image_a_b_gen(batch_size),
	samples_per_epoch=1000,
	nb_epoch=15)

# Test model
print model.evaluate(Xtest, Ytest, batch_size=batch_size)
output = model.predict(Xtest)

# Output colorizations
for i in range(len(output)):
	cur = np.zeros((128, 128, 3))
	cur[:,:,0] = Xtest[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave("colorizations/img_"+str(i)+".png", lab2rgb(cur))
	imsave("colorizations/img_gray_"+str(i)+".png", rgb2gray(lab2rgb(cur)))
