import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard 
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf

# Get images
X = []
for filename in os.listdir('/data/images/Train/'):
    X.append(img_to_array(load_img('/data/images/Train/'+filename)))
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X

#Load weights
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()

def conv_stack(data, filters, s):
        output = Conv2D(filters, (3, 3), strides=s, activation='relu', padding='same')(data)
        #output = BatchNormalization()(output)
        return output

embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = conv_stack(encoder_input, 64, 2)
encoder_output = conv_stack(encoder_output, 128, 1)
encoder_output = conv_stack(encoder_output, 128, 2)
encoder_output = conv_stack(encoder_output, 256, 1)
encoder_output = conv_stack(encoder_output, 256, 2)
encoder_output = conv_stack(encoder_output, 512, 1)
encoder_output = conv_stack(encoder_output, 512, 1)
encoder_output = conv_stack(encoder_output, 256, 1)

#Fusion
fusion_output = RepeatVector(32 * 32)(embed_input) 
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = concatenate([fusion_output, encoder_output], axis=3) 
fusion_output = Conv2D(256, (1, 1), activation='relu')(fusion_output) 

#Decoder
decoder_output = conv_stack(fusion_output, 128, 1)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = conv_stack(decoder_output, 64, 1)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = conv_stack(decoder_output, 32, 1)
decoder_output = conv_stack(decoder_output, 16, 1)
decoder_output = Conv2D(2, (2, 2), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)


#Create embedding
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)

#Generate training data
batch_size = 10

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

#Train model
filepath="weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_weights_only=True, period=20)
callbacks_list = [checkpoint]
model.compile(optimizer='adam', loss='mse')
model.fit_generator(image_a_b_gen(batch_size), epochs=5, steps_per_epoch=1, callbacks=callbacks_list, verbose=1)

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("color_tensorflow_end.h5")

#Make predictions on validation images
color_me = []
for filename in os.listdir('/data/images/Test/'):
    color_me.append(img_to_array(load_img('/data/images/Test/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me_embed = create_inception_embedding(color_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))
