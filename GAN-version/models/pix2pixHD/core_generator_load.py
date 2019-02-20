from keras.models import model_from_json, Model
from keras.layers import Conv2D, Lambda, add, AvgPool2D, Activation, UpSampling2D, Input, concatenate, Reshape, Flatten, Dense
from .utils.conv2d_r import Conv2D_r
from keras.utils import multi_gpu_model
from .utils.instance_normalization import InstanceNormalization


class CoreGenerator():
    """Core Generator.
        
    # Arguments
        width: Width of image in pixels
        height: Height of image in pixels
        channels: Channels for the input image and the generated image
        gpus: The number of gpus you will be using. 
    """

    def __init__(self,
                 width=384,
                 height=384,
                 channels=1,
                 gpus = 0):
        
        self.width = width
        self.height = height
        self.input_channels = channels 
        self.channels = channels
        self.gpus = gpus
        
        # -----------------------
        #  Core Generator Encoder
        # -----------------------

        core_generator_idea = Input(shape=(self.width, self.height, self.input_channels,))
        core_generator_idea_downsample = AvgPool2D(2, padding='same')(core_generator_idea)
        
        core_generator_style = Input(shape=(self.width/(2**7), self.height/(2**7), self.input_channels,))
        
        # -----------------------
        #  Idea Head
        # -----------------------
        
        
        encoder = Conv2D_r(64, 7, 1, core_generator_idea_downsample)
        encoder = InstanceNormalization(axis=-1)(encoder)
        encoder = Activation('relu')(encoder)

        encoder = Conv2D_r(128, 3, 2, encoder)
        encoder = InstanceNormalization(axis=-1)(encoder)
        encoder = Activation('relu')(encoder)

        encoder = Conv2D_r(256, 3, 2, encoder)
        encoder = InstanceNormalization(axis=-1)(encoder)
        encoder = Activation('relu')(encoder)

        encoder = Conv2D_r(512, 3, 2, encoder)
        encoder = InstanceNormalization(axis=-1)(encoder)
        encoder = Activation('relu')(encoder)

        encoder = Conv2D_r(512, 3, 2, encoder)
        encoder = InstanceNormalization(axis=-1)(encoder)
        encoder = Activation('relu')(encoder)
        
        # -----------------------
        #  Style Head
        # -----------------------
        
        
        style = Conv2D_r(128, 3, 1, core_generator_style)
        style = InstanceNormalization(axis=-1)(style)
        style = Activation('relu')(style)
        
        style = UpSampling2D(2)(style)
        style = Conv2D_r(256, 3, 1, style)
        style = InstanceNormalization(axis=-1)(style)
        style = Activation('relu')(style)
        
        style = UpSampling2D(2)(style)
        style = Conv2D_r(512, 3, 1, style)
        style = InstanceNormalization(axis=-1)(style)
        style = Activation('relu')(style)
     
       
        # -----------------------
        #  Merge Style and Idea
        # -----------------------
        
        style_and_idea = concatenate([encoder, style], axis=-1)
        style_and_idea = Conv2D_r(1024, 3, 1,  style_and_idea)
        style_and_idea = InstanceNormalization(axis=-1)(style_and_idea)
        style_and_idea = Activation('relu')(style_and_idea)
        
        style_and_idea = Conv2D_r(512, 3, 1,  style_and_idea)
        style_and_idea = InstanceNormalization(axis=-1)( style_and_idea)
        style_and_idea = Activation('relu')(style_and_idea)
        
        # -------------------------------
        #   Core Generator Residual Block
        # -------------------------------

        def ResidualUnit(input_features):
            output_features = Conv2D_r(512, 3, 1, input_features)
            output_features = InstanceNormalization(axis=-1)(output_features) 
            output_features = Activation('relu')(output_features)
            output_features = Conv2D_r(512, 3, 1, output_features)
            output_features = InstanceNormalization(axis=-1)(output_features)
            output_features = add([input_features, output_features])
            output_features = Activation('relu')(output_features)
            return output_features

        resnet = ResidualUnit(style_and_idea)
        resnet = ResidualUnit(resnet)
        resnet = ResidualUnit(resnet)
        resnet = ResidualUnit(resnet)
        resnet = ResidualUnit(resnet)
        resnet = ResidualUnit(resnet)
        resnet = ResidualUnit(resnet)
        resnet = ResidualUnit(resnet)
        resnet = ResidualUnit(resnet)

        
        # -------------
        #  Core Decoder
        # -------------

        decoder = UpSampling2D(2)(resnet)
        decoder = Conv2D_r(512, 3, 1, decoder)
        decoder = InstanceNormalization(axis=-1)(decoder)
        decoder = Activation('relu')(decoder)

        decoder = UpSampling2D(2)(decoder)
        decoder = Conv2D_r(256, 3, 1, decoder)
        decoder = InstanceNormalization(axis=-1)(decoder)
        decoder = Activation('relu')(decoder)

        decoder = UpSampling2D(2)(decoder)
        decoder = Conv2D_r(128, 3, 1, decoder)
        decoder = InstanceNormalization(axis=-1)(decoder)
        decoder = Activation('relu')(decoder)

        decoder = UpSampling2D(2)(decoder)
        decoder = Conv2D_r(64, 3, 1, decoder)
        features = Lambda(lambda x: x, name='core_features_org')(decoder)
        decoder = InstanceNormalization(axis=-1)(decoder)
        decoder = Activation('relu')(decoder)

        decoder = Conv2D_r(channels, 7, 1, decoder)
        picture_lowres = Activation('tanh')(decoder)
        
        core_generator = Model([core_generator_idea, core_generator_style], [picture_lowres, features])
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
