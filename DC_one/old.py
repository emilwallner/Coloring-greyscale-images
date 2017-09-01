old.py
# model = Sequential()
# model.add(InputLayer(input_shape=(None, None, 1)))
# model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
# model.add(UpSampling2D((2, 2)))
#     return model


# # Set up model
# N = 5
# model = Sequential()
# num_maps1 = [4, 8, 16, 32, 64]
# num_maps2 = [8, 16, 32, 64, 128]


# model = tflearn.DNN(net)




# # Set up model
# model = Sequential()
# num_maps1 = [64, 32, 16, 8, 4]
# num_maps2 = [8, 16, 32, 64, 128]

# # Convolutional layers

# model.add(Conv2D(4, 3, 3, border_mode='same', activation='relu', input_shape=(128, 128, 1)))
# for i in range(4):
# 	model.add(Conv2D(num_maps1[i], 3, 3, border_mode='same', activation='relu', strides=(1, 1)))
# 	model.add(BatchNormalization())
# 	model.add(Conv2D(num_maps2[i], 3, 3, border_mode='same', activation='relu', strides=(2, 2)))
# 	model.add(BatchNormalization())

# # Upsampling layers
# for i in range(5):
# 	model.add(UpSampling2D(size=(2, 2)))
# 	model.add(Convolution2D(num_maps2[-(i+1)], 3, 3, border_mode='same'))
# 	model.add(BatchNormalization())
# 	model.add(Activation('relu'))
# 	if i != N-1:
# 		model.add(Convolution2D(num_maps1[-(i+1)], 3, 3, border_mode='same'))
# 		model.add(BatchNormalization())
# 		model.add(Activation('relu'))
# 	else:
# 		model.add(Convolution2D(2, 3, 3, border_mode='same'))









