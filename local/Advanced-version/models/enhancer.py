from keras.models import model_from_json, Model
from keras.layers import Conv2D, Lambda, add, AvgPool2D, Activation, UpSampling2D, Input, concatenate, Reshape
from .utils.conv2d_r import Conv2D_r
from .utils.instance_normalization import InstanceNormalization
from keras.utils import multi_gpu_model



class Enhancer():
    """Enhancer.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
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
    
        # ---------------------------
        #  Enhancer Generator Encoder
        # ---------------------------

        enhancer_generator_input = Input(shape=(self.width, self.height, channels,))
        enhancer_core_features = Input(shape=(self.width/2, self.height/2, 64,))

        encoder = Conv2D_r(32, 7, 1, enhancer_generator_input)
        encoder = InstanceNormalization(axis=-1)(encoder)
        encoder = Activation('relu')(encoder)

        encoder = Conv2D_r(64, 3, 2, encoder)
        enhancer_and_core = concatenate([encoder, enhancer_core_features], axis=-1)
        enhancer_and_core = InstanceNormalization(axis=-1)(enhancer_and_core)
        enhancer_and_core = Activation('relu')(enhancer_and_core)
        
        enhancer_and_core = Conv2D_r(64, 3, 1,  enhancer_and_core)
        enhancer_and_core = InstanceNormalization(axis=-1)(enhancer_and_core)
        enhancer_and_core = Activation('relu')(enhancer_and_core)
        
        
        # ----------------------------------
        #  Enhancer Generator Residual Block
        # ----------------------------------

        def ResidualUnitLocal(input_features):
            output_features = Conv2D_r(64, 3, 1, input_features) 
            output_features = InstanceNormalization(axis=-1)(output_features) 
            output_features = Activation('relu')(output_features)
            output_features = Conv2D_r(64, 3, 1, output_features)
            output_features = InstanceNormalization(axis=-1)(output_features) 
            output_features = add([input_features, output_features])
            output_features = Activation('relu')(output_features)
            return output_features

        resnet = ResidualUnitLocal(enhancer_and_core)
        resnet = ResidualUnitLocal(resnet)
        resnet = ResidualUnitLocal(resnet)

        # ---------------------------
        #  Enhancer Generator Decoder
        # ---------------------------

        decoder = UpSampling2D(2)(resnet)
        decoder = Conv2D_r(64, 3, 1, decoder)
        decoder = InstanceNormalization(axis=-1)(decoder)
        decoder = Activation('relu')(decoder)

        decoder = Conv2D_r(channels, 7, 2, decoder)
        enhanced_picture = Activation('tanh')(decoder)

        # -----------------
        #  Save model
        # -----------------
        
        if self.gpus < 2:
            self.model = Model([enhancer_generator_input, enhancer_core_features], enhanced_picture) 
            self.save_model = self.model
        else:
            self.save_model = Model([enhancer_generator_input, enhancer_core_features], enhanced_picture) 
            self.model = multi_gpu_model(self.save_model, gpus=gpus)
