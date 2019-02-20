# System Modules
import os
import time

# Custom Modules
from models.utils.calc_output_and_feature_size import calc_output_and_feature_size
from lib.data_utils import save_sample_images, write_log, generate_training_images
from lib.data_utils import generator, turn_to_rgb_384, generate_label_data, resize_images
from models.discriminator_full import DiscriminatorFull
from models.discriminator_low import DiscriminatorLow
from models.core_generator_load import CoreGenerator
from models.style_features import StyleFeatures
from models.enhancer import Enhancer

# Keras Modules
from keras.utils import multi_gpu_model
from keras.layers import Lambda, UpSampling2D, Input, concatenate
from keras.utils.data_utils import  GeneratorEnqueuer
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.models import Model, save_model, load_model
from keras import backend as K
K.clear_session()
import keras
print(keras.backend.floatx())

# Import models
from models.discriminator_full import DiscriminatorFull
from models.discriminator_low import DiscriminatorLow
from models.core_generator_load import CoreGenerator
from models.style_features import StyleFeatures
from models.enhancer import Enhancer

# Other Moduals 
import tensorflow as tf
import numpy as np
from numba import jit

# ----------
#  Settings 
# ----------

height = 384
width = 384
channels = 1
epochs = 2
gpus = 1
batch_size = 8 * 4
cpus = 10
use_multiprocessing = True
save_weights_every_n_epochs = 0.01
max_queue_size=batch_size * 3
img_dir = "../data/drawings_384/"
resource_dir = "./resources/"
dataset_len = len(os.listdir(img_dir))
learning_rate = 0.0002
experiment_name = time.strftime("%Y-%m-%d-%H-%M")
decay_rate = learning_rate / ((dataset_len / batch_size) * epochs)

# ----------------------------------
#  Create directory for sample data
# ----------------------------------

main_dir = './output/enhancer/' + experiment_name
save_sample_images_dir = main_dir + '/sample_images/'
weights_dir = main_dir +'/weights/'
log_path = main_dir + '/logs/'
model_path = main_dir + '/models/'

if not os.path.exists(main_dir):
    os.makedirs(main_dir)
    os.makedirs(save_sample_images_dir)
    os.makedirs(log_path)
    os.makedirs(weights_dir)
    os.makedirs(model_path)


# ---------------
#  Import Models 
# ---------------

@jit
def concatenateNumba(x, y):
    return np.concatenate([x, y], axis=-1)

# --------------------------------
#  Create GAN with core generator
# --------------------------------

style_features = StyleFeatures(gpus=gpus)
core_generator = CoreGenerator(gpus=gpus)
enhancer = Enhancer(gpus=gpus)
discriminator_full = DiscriminatorFull(gpus=gpus, decay_rate=decay_rate)
discriminator_low = DiscriminatorLow(gpus=gpus, decay_rate=decay_rate)

style_features.model.load_weights(resource_dir + 'style_features.h5')
core_generator.model.load_weights(resource_dir + 'core_generator.h5')
enhancer.model.load_weights(resource_dir + 'enhancer.h5')
discriminator_full.model.load_weights(resource_dir + 'discriminator_full.h5')
discriminator_low.model.load_weights(resource_dir + 'discriminator_low.h5')

discriminator_full.model.trainable = False
discriminator_low.model.trainable = False

# Generate image with core generator
gan_x = Input(shape=(height, width, channels,))
gan_y = Input(shape=(height, width, channels,))

# Extract style features and add them to image
gan_y_features = style_features.model(gan_y)

gan_lowres, gan_features = core_generator.model([gan_x, gan_y_features])
gan_lowres = UpSampling2D(2)(gan_lowres)
enhance_input = concatenate([gan_x, gan_lowres], axis=-1)
enhanced_picture = enhancer.model([enhance_input, gan_features])

# Extract features and predictions from discriminators
disc_input = concatenate([gan_x, enhanced_picture], axis=-1)
gan_predictions_full, disc_fake_features_full = discriminator_full.model(disc_input)
gan_predictions_low, disc_fake_features_low = discriminator_low.model(disc_input)

# Compile GAN
gan_enhanced_org = Model(inputs=[gan_x, gan_y], outputs=[enhanced_picture, disc_fake_features_full, disc_fake_features_low, gan_predictions_full, gan_predictions_low])                  

gan_enhanced_org.name = "gan_enhanced"
optimizer = Adam(learning_rate, 0.5, decay=decay_rate)
loss_gan = ['mae', 'mae', 'mae','mse', 'mse']
loss_weights_gan = [1, 10, 10, 0.5, 0.5]

gan_enhanced = multi_gpu_model(gan_enhanced_org)
gan_enhanced.compile(optimizer=optimizer, loss_weights=loss_weights_gan, loss=loss_gan)


# --------------------------------
#  Compile Discriminator
# --------------------------------

discriminator_full.model.trainable = True
discriminator_low.model.trainable = True

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)

loss_d = ['mse', zero_loss]
loss_weights_d = [0.5, 0]

discriminator_full_multi = multi_gpu_model(discriminator_full.model)
discriminator_low_multi = multi_gpu_model(discriminator_low.model, gpus=2)
discriminator_full_multi.compile(optimizer=optimizer, loss_weights=loss_weights_d, loss=loss_d)
discriminator_low_multi.compile(optimizer=optimizer, loss_weights=loss_weights_d, loss=loss_d)


# --------------------------------------------------
#  Import all the filenames for the training images
# --------------------------------------------------

X = []
for filename in os.listdir(img_dir):
        filepath = os.path.join(img_dir, filename)
        X.append([filepath, filename])

# --------------------------------------------------
#  Initiate Generator Queue
# --------------------------------------------------

enqueuer = GeneratorEnqueuer(generator(X, batch_size, dataset_len, width, height), use_multiprocessing=use_multiprocessing, wait_time=0.01)
enqueuer.start(workers=cpus, max_queue_size=max_queue_size)
output_generator = enqueuer.get()


# ---------------------------------
#  Initiate values for Tensorboard
# ---------------------------------

callback_disc = TensorBoard(log_path)
callback_disc_low = TensorBoard(log_path)
callback_gan = TensorBoard(log_path)

callback_disc.set_model(discriminator_full.save_model)
callback_disc_low.set_model(discriminator_low.save_model)
callback_gan.set_model(gan_enhanced_org)

callback_disc_names = ['weighted_loss_real_full', 'disc_loss_real_full', 'zero_1', 'weighted_loss_fake_full', 'disc_loss_fake_full', 'zero_2']
callback_disc_low_names = ['weighted_loss_real_low', 'disc_loss_real_low', 'zero_3', 'weighted_loss_fake_low', 'disc_loss_fake_low', 'zero_4']
callback_gan_names = ['total_gan_loss', 'image_diff', 'feature_diff_disc_full', 'feature_diff_disc_low', 'predictions_full', 'predictions_low']

# Decide how often to create sample images, save log data, and weights. 
cycles = int(epochs * (dataset_len / batch_size))
save_images_cycle = int((dataset_len / batch_size) / 50)
save_weights_cycle = int((dataset_len / batch_size) * save_weights_every_n_epochs)

# Calculate the discriminator output size for features and image predictions
pred_size_f, feat_size_f = calc_output_and_feature_size(width, height)
pred_size_l, feat_size_l = calc_output_and_feature_size(width/2, height/2)

# Create benchmark to see progress
x_benchmark, y_benchmark, _ = next(output_generator)
start = time.time()

for i in range(0, cycles):
    start_c = time.time()
    # ------------------------
    #  Generate Training Data
    # ------------------------

    # Discriminator data
    x_full, y_full, x_and_y_full = next(output_generator)
    x_low, y_low, x_and_y_low = next(output_generator)
    
    # Fixed data
    fake_labels_f, true_labels_f, dummy_f = generate_label_data(batch_size, pred_size_f, feat_size_f)
    fake_labels_l, true_labels_l, dummy_l = generate_label_data(batch_size, pred_size_l, feat_size_l)
  
    # GAN data
    x_gan, y_gan, x_and_y_gan = next(output_generator)

    # ----------------------
    #  Train Discriminators 
    # ----------------------

    y_gen_full, _, _, _, _ = gan_enhanced.predict([x_full, y_full])
    x_and_y_gen_full = concatenateNumba(x_full, y_gen_full)
    
    # Prepare data for Low Resolution Discriminator 
    y_gen_low, _, _, _, _ = gan_enhanced.predict([x_low, y_low])
    x_and_y_gen_low = concatenateNumba(x_low, y_gen_low)

    # Train Discriminators 
    d_loss_fake_full = discriminator_full_multi.train_on_batch(x_and_y_gen_full, [fake_labels_f, dummy_f])
    d_loss_real_full = discriminator_full_multi.train_on_batch(x_and_y_full, [true_labels_f, dummy_f])
   
    d_loss_fake_low = discriminator_low_multi.train_on_batch(x_and_y_gen_low, [fake_labels_l, dummy_l])
    d_loss_real_low = discriminator_low_multi.train_on_batch(x_and_y_low, [true_labels_l, dummy_l])

    # -----------
    #  Train GAN
    # -----------
    

    # Extract featuers from discriminators 
    _, real_features_full = discriminator_full_multi.predict(x_and_y_gan)
    _, real_features_low = discriminator_low_multi.predict(x_and_y_gan)
    
    # Train GAN on one batch
    gan_core_loss = gan_enhanced.train_on_batch([x_gan, y_gan], [y_gan, 
                                                            real_features_full, 
                                                            real_features_low,
                                                            true_labels_f,
                                                            true_labels_l])

    # -------------------------------------------
    #  Save image samples, weights, and log data
    # -------------------------------------------
    
    # Print log data to tensorboard
    write_log(callback_disc, callback_disc_names, d_loss_fake_full + d_loss_real_full, i)
    write_log(callback_disc_low, callback_disc_low_names, d_loss_fake_low + d_loss_real_low, i)
    write_log(callback_gan, callback_gan_names, gan_core_loss, i)
    
    end_c = time.time()
    print("\n\nCycle:", i)
    print("Time:", end_c - start_c)
    print("Total images:", batch_size * i)

    # Save sample images
    if i % save_images_cycle == 0:
        print('Print those bad boys:', i)
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        output_benchmark, _, _, _, _ = gan_enhanced.predict([x_benchmark, y_benchmark])
        save_sample_images(output_benchmark, x_benchmark, y_benchmark, 'b-' + str(i), save_sample_images_dir)
        save_sample_images(y_gen_full, x_full, y_full, str(i), save_sample_images_dir)
        start = time.time()

    #  Save weights
    if i % save_weights_cycle == 0:
        discriminator_full.save_model.save_weights(weights_dir + str(i) + "-discriminator_full.h5")
        discriminator_low.save_model.save_weights(weights_dir + str(i) + "-discriminator_low.h5")
        enhancer.save_model.save_weights(weights_dir + str(i) + "-enhancer.h5")
        core_generator.save_model.save_weights(weights_dir + str(i) + "-core_generator.h5")
        style_features.save_model.save_weights(weights_dir + str(i) + "-style_features.h5")
        
        discriminator_full.save_model.save_weights(resource_dir + "discriminator_full.h5")
        discriminator_low.save_model.save_weights(resource_dir + "discriminator_low.h5")
        enhancer.save_model.save_weights(resource_dir + "enhancer.h5")
        core_generator.save_model.save_weights(resource_dir + "core_generator.h5")
        style_features.save_model.save_weights(resource_dir + "style_features.h5")

