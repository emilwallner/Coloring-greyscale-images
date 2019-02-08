from keras.models import Model, save_model, load_model
from keras.optimizers import Adam
from .utils.conv2d_r import Conv2D_r
from keras.utils import multi_gpu_model
from .utils.instance_normalization import InstanceNormalization
import tensorflow as tf
from keras import backend as K
from .utils.sn import ConvSN2D, DenseSN

def zero_loss(y_true, y_pred):
            return K.zeros_like(y_true)

class CoreGeneratorEnhancer():
    """Core Generator.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
        gpus: The number of gpus you will be using. 
    """

    def __init__(self,
                 resource_path='./resources/',
                 gpus = 0):
        
        self.gpus = gpus
        
        core_generator_original = load_model(resource_path + 'core_generator.h5', custom_objects={'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        core_generator = Model(inputs=core_generator_original.input, outputs=[core_generator_original.output, core_generator_original.get_layer('core_features_org').output])
        core_generator.name = "core_generator"
        core_generator.trainable = True
	
        self.model = core_generator        
        self.save_model = core_generator 
    
class CoreGenerator():
    """Core Generator.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
        gpus: The number of gpus you will be using. 
    """

    def __init__(self,
                 resource_path='./resources/',
                 gpus = 0):
        
        self.gpus = gpus
        
        core_generator = load_model(resource_path + 'core_generator.h5', custom_objects={'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        #core_generator = Model(inputs=core_generator_original.input, 
        #                    outputs=[core_generator_original.get_layer('core_features_org').output, # core_generator_original.get_layer('core_features_true').output])
        core_generator.name = "core_generator"
        core_generator.trainable = True
	        
        self.model = core_generator
        self.save_model = core_generator
            
class Enhancer():
    """Enhancer.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
        gpus: The number of gpus you will be using. 
    """

    def __init__(self,
                 resource_path='./resources/',
                 gpus = 0):
        
        self.gpus = gpus
        
        enhancer = load_model(resource_path + 'enhancer.h5', custom_objects={'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        enhancer.name = 'enhancer'
        enhancer.trainable = True
        
        self.model = enhancer
        self.save_model = enhancer

class DiscriminatorFull():
    """Core Generator.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
        gpus: The number of gpus you will be using.
        learning_rate: learning rate
        decay_rate: 
    """

    def __init__(self,
                 resource_path='./resources/',
                 learning_rate=0.0002,
                 decay_rate=2e-6,
                 gpus = 1):
        
        self.gpus = gpus
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        

        def zero_loss(y_true, y_pred):
        	return K.zeros_like(y_true)
               
        discriminator_full = load_model(resource_path + 'discriminator_full.h5', custom_objects={'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'zero_loss': zero_loss, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        
        discriminator_full.trainable = True
        discriminator_full.name = "discriminator_full"
        
        self.model = discriminator_full
        self.save_model = discriminator_full
        
        
class DiscriminatorLow():
    """Core Generator.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
        gpus: The number of gpus you will be using. 
        learning_rate: learning rate
        decay_rate: 
    """

    def __init__(self,
                 resource_path='./resources/',
                 learning_rate=0.0002,
                 decay_rate=2e-6,
                 gpus = 0):
        
        self.gpus = gpus
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        def zero_loss(y_true, y_pred):
        	return K.zeros_like(y_true)
       
        discriminator_low = load_model(resource_path + 'discriminator_low.h5', custom_objects={'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf,'zero_loss': zero_loss, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        
        discriminator_low.trainable = True
        discriminator_low.name = "discriminator_low"

        self.model = discriminator_low
        self.save_model = discriminator_low  

class StyleFeatures():
    """Core Generator.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
        gpus: The number of gpus you will be using. 
        learning_rate: learning rate
        decay_rate: 
    """

    def __init__(self,
                 resource_path='./resources/',
                 gpus = 0):
        
        self.gpus = gpus
        
        style_features = load_model(resource_path + 'style_features.h5', custom_objects={'Conv2D_r': Conv2D_r, 'InstanceNormalization': InstanceNormalization, 'tf': tf, 'ConvSN2D': ConvSN2D, 'DenseSN': DenseSN})
        
        style_features.trainable = True
        style_features.name = "style_features"
        
        self.model = style_features
        self.save_model = style_features

