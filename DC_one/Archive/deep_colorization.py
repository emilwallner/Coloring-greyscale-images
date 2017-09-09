from keras.layers import Conv2D, UpSampling2D, InputLayer
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

# Get images
X = img_to_array(load_img('pinkfloyd.jpg'))
X = np.array(X)

# Set up train and test data
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]


def conv_stack(filters, d, strides):
	for i in strides:
		model.add(Conv2D(filters, (3, 3), strides=i, activation='relu', dilation_rate=d, padding='same'))

def upsampling_stack(filters):
	for i in filters:
		model.add(UpSampling2D((2, 2)))
		conv_stack(i, 1, [1, 1])

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))

conv_stack(4, 1, [1, 2])
conv_stack(8, 1, [1, 2])
conv_stack(16, 1, [1, 1, 2])
conv_stack(32, 2, [1, 1, 1, 1, 1, 1])
conv_stack(32, 1, [1, 1, 1])
upsampling_stack([16, 8, 4])
conv_stack(2, 1, [1, 1])

# Finish model
model.compile(optimizer='rmsprop',
			loss='mse')

# Generate training data
X = rgb2lab(X[:,:,:,0])
Y = lab[:,:,:,1:]


# Train model
model.fit(x=X, 
	y=Y,
	batch_size=1,
	epochs=1000)

# Test model
print(model.evaluate(X, Y, batch_size=1))
output = model.predict(X)

# Output colorizations
for i in range(len(output)):
	cur = np.zeros((128, 128, 3))
	cur[:,:,0] = Xtest[i][:,:,0]
	cur[:,:,1:] = output[i]
	imsave("colorizations/img_"+str(i)+".png", lab2rgb(cur))
	imsave("colorizations/img_gray_"+str(i)+".png", rgb2gray(lab2rgb(cur)))
