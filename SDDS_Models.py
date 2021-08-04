import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Add, SeparableConv2D, MaxPooling2D, Activation, BatchNormalization, Conv2D



def MP_CNN( benchmark= 'NYUV2', weights_path=''):   
    if benchmark == 'NYUV2':
        input_shape=(480,640,3)
    if benchmark == 'CITYSCAPES':
        input_shape=(512,1024,3)
    
    input = Input(shape = input_shape)
    x = MaxPooling2D()(input)

    x = Conv2D( 16,3 ,padding='same', activation= 'relu')(x)
    x = Conv2D( 16,3 ,padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D( 64,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 64,3, padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D( 1024,1, padding='same', activation= 'linear')(x)
    out = tf.nn.depth_to_space(x, 32)

    model = Model(input, out)
    model.load_weights(weights_path)
    return model
    
def strided_Conv_CNN(benchmark= 'NYUV2', weights_path=''):
    if benchmark == 'NYUV2':
        input_shape=(480,640,3)
    if benchmark == 'CITYSCAPES':
        input_shape=(512,1024,3)
        
    input = Input(shape = input_shape)
    x = Conv2D( 16,3, strides= (2,2),padding='same', activation= 'relu')(input)
    
    x = Conv2D( 16,3 ,padding='same', activation= 'relu')(x)
    x = Conv2D( 16,3, strides= (2,2), padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D( 64,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 64,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 64,3, strides= (2,2), padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 256,3, strides= (2,2), padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 1024,3, strides= (2,2), padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)

    x = tf.keras.layers.Conv2D( 1024,1, padding='same', activation= 'linear')(x)
    out = tf.nn.depth_to_space(x, 32)

    model = Model(input, out)
    model.load_weights(weights_path)
    return model
    
def SDDS_Net(benchmark= 'NYUV2', sim = False, weights_path=''):
    
    if benchmark == 'NYUV2':
        input_shape=(480,640,3)
    if benchmark == 'CITYSCAPES':
        input_shape=(512,1024,3)
        
    input = Input(shape = input_shape)
    x = tf.nn.space_to_depth(input, 2)

    x = Conv2D( 16,3 ,padding='same', activation= 'relu')(x)
    x = Conv2D( 16,3 ,padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = tf.nn.space_to_depth(x, 2)

    x = Conv2D( 64,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 64,3, padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = tf.nn.space_to_depth(x, 2)

    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 256,3, padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = tf.nn.space_to_depth(x, 2)

    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 1024,3, padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)
    x = tf.nn.space_to_depth(x, 2)

    x1 = tf.keras.layers.Conv2D( 1024,1, padding='same', activation= 'linear')(x)
    out = tf.nn.depth_to_space(x1, 32)
    
    if sim:
        x2 = tf.keras.layers.Conv2D( 1024,1, padding='same', activation= 'linear')(x)
        out2 = tf.nn.depth_to_space(x, 32)
        model = Model(input, [out, out2])
    else:
        model = Model(input, out)
    model.load_weights(weights_path)
    return model

    
def SDDS_Net_DW(benchmark= 'NYUV2', sim= False, weights_path=''):
    
    if benchmark == 'NYUV2':
        input_shape=(480,640,3)
    if benchmark == 'CITYSCAPES':
        input_shape=(512,1024,3)
    
    input = Input(shape= input_shape)
    x = tf.nn.space_to_depth(input, 2)

    x = Conv2D( 16,3, padding='same', activation= 'relu')(x)
    x = Conv2D( 16,3, padding='same', activation= 'relu')(x)
    x = BatchNormalization()(x)

    x2 = tf.nn.space_to_depth(x, 2)

    x = Activation('relu')(x2)
    x = SeparableConv2D( 64,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 64,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 64,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 64,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 64,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 64,3, padding='same')(x)
    x = BatchNormalization()(x)
    add = Add()([x2,x])

    x3 = tf.nn.space_to_depth(add, 2)

    x = Activation('relu')(x3)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    add2 = Add()([x3,x])

    x = Activation('relu')(add2)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    add3 = Add()([add2,x])

    x = Activation('relu')(add3)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    add4 = Add()([add3,x])

    x = Activation('relu')(add4)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 256,3, padding='same')(x)
    x = BatchNormalization()(x)
    add5 = Add()([add4,x])

    x4 = tf.nn.space_to_depth(add5, 2)

    x = Activation('relu')(x4)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D( 1024,3, padding='same')(x)
    x = BatchNormalization()(x)
    add6 = Add()([x4,x])

    x = tf.nn.space_to_depth(add6, 2)

    x1 = tf.keras.layers.Conv2D( 1024,1, padding='same', activation= 'linear')(x)
    out = tf.nn.depth_to_space(x1, 32)
    
    if sim:
        x2 = tf.keras.layers.Conv2D( 1024,1, padding='same', activation= 'linear')(x)
        out2 = tf.nn.depth_to_space(x, 32)
        model = Model(input, [out, out2])
    else:
        model = Model(input, out)
    model.load_weights(weights_path)
    return model