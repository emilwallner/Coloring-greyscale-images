from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers import LeakyReLU, Concatenate, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from .utils.instance_normalization import InstanceNormalization
from .utils.sn import ConvSN2D
from .utils.attention import Attention
import keras


class CoreGenerator():
    """Core Generator.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image
        gpus: The number of gpus you will be using. 
    """

    def __init__(self,
                 width=256,
                 height=256,
                 channels=1,
                 gpus = 0):
        
        self.width = width
        self.height = height
        self.channels = channels
        self.gpus = gpus
        self.gf = 64

        
        # -------------------------------------------------------------------------------------
        #  Core Generator 
        #  The U-net structure is from Erik Linder-Noren's brilliant pix2pix model
        #  Source: https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
        #  Modifications: Thinner to enable 128x128 images, Spectral Normalization and 
        #  an Attention layer. 
        # -------------------------------------------------------------------------------------
        

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = ConvSN2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)

            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = ConvSN2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d1 = Input(shape=(width, height, channels))

        # Downsampling
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u4_att = Attention(512)(u4)
        u5 = deconv2d(u4_att, d2, self.gf*2)

        u6 = UpSampling2D(size=2)(u5)
        output = ConvSN2D(2, kernel_size=(7,7), strides=1, padding='same', activation='tanh')(u6)
        
        core_generator = Model(d1, output)
        core_generator.name = "core_generator"
        
        # --------------
        #  Compile Model
        # --------------
        
        if self.gpus < 2:
            self.model = core_generator
            self.save_model = self.model
        else:
            self.save_model = core_generator
            self.model = multi_gpu_model(self.save_model, gpus=gpus)
