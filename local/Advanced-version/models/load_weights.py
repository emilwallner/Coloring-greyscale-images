import keras
from keras.models import Model, save_model, load_model
from core_generator_load import CoreGenerator
from discriminator_full import DiscriminatorFull
from discriminator_low import DiscriminatorLow
from style_features import StyleFeatures
from enhancer import Enhancer

from keras.optimizers import Adam
from keras.models import model_from_json
from utils.conv2d_r import Conv2D_r
from keras.utils import multi_gpu_model
from utils.instance_normalization import InstanceNormalization
import tensorflow as tf
from keras import backend as K
from utils.sn import ConvSN2D, DenseSN
from keras.models import Model, save_model, load_model

def zero_loss(y_true, y_pred):
            return K.zeros_like(y_true)


style_features = StyleFeatures(gpus=1)
core_generator = CoreGenerator(gpus=1)
discriminator_full = DiscriminatorFull(gpus=1, decay_rate=0)
discriminator_low = DiscriminatorLow(gpus=1, decay_rate=0)
enhancer = Enhancer(gpus=1)

resource_path='./weights/'
save_path = './resources/'
learning_rate=0.0002,
decay_rate=0

core_generator.model.load_weights(resource_path + "core_generator.h5")
enhancer.model.load_weights(resource_path + 'enhancer.h5')
discriminator_full.model.load_weights(resource_path + 'discriminator_full.h5')
discriminator_low.model.load_weights(resource_path + 'discriminator_low.h5')
style_features.model.load_weights(resource_path + 'style_features.h5')        

     
save_model(discriminator_full.model, save_path + "discriminator_full.h5")
save_model(discriminator_low.model, save_path + "discriminator_low.h5")
save_model(enhancer.model, save_path + "enhancer.h5")
save_model(core_generator.model, save_path + "core_generator.h5")
save_model(style_features.model, save_path + "style_features.h5")   