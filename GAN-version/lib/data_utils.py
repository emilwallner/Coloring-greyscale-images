import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.preprocessing import image
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense
from keras.models import Model
import os
from skimage.transform import resize, rotate, rescale
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import random


def turn_filename_into_image(filenames, batch_size, width, height, img_dir):
    
    empty_x = []
    rot_value = random.randint(-20,20)
    flip_lr = bool(random.getrandbits(1))
    flip_ud = bool(random.getrandbits(1))
    
    for name in filenames:
        image_x = img_to_array(load_img(img_dir + name, target_size=(width, height)))
        image_x = np.array(image_x, dtype='float32')
        image_x = (1.0/(255./2))*image_x - 1
     
        image_x = rotate(image_x, rot_value, mode='reflect')

        if flip_lr:
            image_x = np.fliplr(image_x)
        
        empty_x.append(image_x)
        
    empty_x = np.array(empty_x, dtype='float32')
    lab_batch = rgb2lab(empty_x)
    X_batch = lab_batch[:,:,:,0] / 100
    X_batch = X_batch.reshape(X_batch.shape+(1,))
    Y_batch = lab_batch[:,:,:,1:] / 128
    
    return np.array(X_batch, dtype='float32'), np.array(Y_batch, dtype='float32')


def random_image_index(dataset_len, batchsize):
    start = random.randint(0,(dataset_len - (batchsize + 1)))
    end = start + batchsize
    return start, end

def generate_training_images(filenames, batch_size, dataset_len, width, height, img_dir):
    
    start, end = random_image_index(dataset_len, batch_size)
    names = filenames[start:end]
    x, y = turn_filename_into_image(names, batch_size, width, height, img_dir)
    x_and_y = np.concatenate([x, y], axis=-1)
    
    return x, y, x_and_y

def generator(X, img_dir, batch_size, dataset_len, width, height):
    while True:
        x, y, x_and_y = generate_training_images(X, batch_size, dataset_len, width, height, img_dir)
        yield x, y, x_and_y

def generate_label_data(batch_size, output_size_pred, output_size_features):

    fake_labels = np.zeros((batch_size, output_size_pred, 1))
    true_labels = np.ones((batch_size, output_size_pred, 1))
    placeholder_input = np.zeros((batch_size, output_size_features, 1))
    
    return fake_labels, true_labels, placeholder_input


def save_each_image(colored_layers, BW_layer, cycle, nr, path, ending):
    
    cur = np.zeros((128, 128, 3))
    cur[:,:,0] = BW_layer[:,:,0] * 100
    cur[:,:,1:] = colored_layers * 128
    imsave(os.path.join(path, cycle + nr + ending), lab2rgb(cur))

def save_sample_images(colored_layers, BW_layer, cycle, path):
    for i in range(len(colored_layers)):
        save_each_image(colored_layers[i], BW_layer[i], cycle, str(i), path, '-gen.png')

        
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
