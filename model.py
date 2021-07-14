import tensorflow as tf

class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=channels, kernel_size=3, padding='same')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=channels, kernel_size=3, padding='same')
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x_input, training = True):

        x_residual = x_input

        x = tf.keras.layers.Activation('relu')(self.batch1(self.conv1(x_input)))
        x = self.batch2(self.conv2(x))

        x = tf.keras.layers.Activation('relu')(self.add([x, x_residual]))

        return x

class UpsamplingConvLayer1(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, upsample_factor):
        super(UpsamplingConvLayer1, self).__init__()
        self.upsample_factor = upsample_factor
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding = 'same')

    def call(self, x):
        shape = tf.shape(x)
        x = tf.image.resize(x, [256, 256])
        x = self.conv2d(x)
        return x

class UpsamplingConvLayer2(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, stride, upsample_factor):
        super(UpsamplingConvLayer2, self).__init__()
        self.upsample_factor = upsample_factor
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding = 'same')

    def call(self, x):
        shape = tf.shape(x)
        x = tf.image.resize(x, [512, 512])
        x = self.conv2d(x)
        return x


class ResNet(tf.keras.Model):

    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=9, padding='same', input_shape = (512, 512, 3))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides = (2, 2))
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides = (2, 2))

        self.res1 = ResidualLayer(128)
        self.res2 = ResidualLayer(128)
        self.res3 = ResidualLayer(128)
        self.res4 = ResidualLayer(128)
        self.res5 = ResidualLayer(128)

        self.deconv1 = UpsamplingConvLayer1(filters=64, kernel_size=3, stride=1, upsample_factor=2)
        self.deconv2 = UpsamplingConvLayer2(filters=32, kernel_size=3, stride=1, upsample_factor=2)
        self.deconv3 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, padding='same')

        self.batch1 = tf.keras.layers.BatchNormalization()
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.batch3 = tf.keras.layers.BatchNormalization()
        self.batch4 = tf.keras.layers.BatchNormalization()
        self.batch5 = tf.keras.layers.BatchNormalization()

    def call(self, x_input, training = True):

        x = tf.keras.layers.Activation('relu')(self.batch1(self.conv1(x_input)))
        x = tf.keras.layers.Activation('relu')(self.batch2(self.conv2(x)))
        x = tf.keras.layers.Activation('relu')(self.batch3(self.conv3(x)))

        x = self.res1(x, training = training)
        x = self.res2(x, training = training)
        x = self.res3(x, training = training)
        x = self.res4(x, training = training)
        x = self.res5(x, training = training)

        x = tf.keras.layers.Activation('relu')(self.batch4(self.deconv1(x)))
        x = tf.keras.layers.Activation('relu')(self.batch5(self.deconv2(x)))
        x = self.deconv3(x)

        return x

    def summary(self):
        x = tf.keras.layers.Input(shape=(512, 512, 3))
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()

class VGG19(tf.keras.models.Model):
    def __init__(self):
        super(VGG19, self).__init__()
        self.content_layers = ['block4_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.num_style_layers = len(self.style_layers)
        self.vgg19 = self._get_model()
    
    def _get_model(self):
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable=False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs

        return tf.keras.models.Model(vgg.input, model_outputs)
    
    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        return self.vgg19(preprocessed_input)


if __name__=="__main__":
    model = ResNet()
    model.compile()

    model.summary()
