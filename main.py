import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image
from functools import partial
from model import ResNet
from preprocessing import Preprocessing
from loss import StyleContentLoss

class NeuralStyleTransfer:

    STYLE_IMAGES = ["Kandinsky_Composition_7", 'Kanagawa']

    def __init__(self, style_image_name):
        self.data_reader = Preprocessing('images', 'preprocessed_images')
        self.target_style_image = self._load_image('style_images/Kandinsky_Composition_7.jpg')
        # self.imshow(self.target_style_image)
        content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
        self.test_content_image = self._load_image(content_path)

    @classmethod
    def _load_image(cls, image_path, max_dim = 512): 
        '''loads and normalises style image to the range [0, 1)'''

        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    @staticmethod
    def _show_image(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        
        plt.imshow(image)
        if title:
            plt.title(title)

        plt.show()

    def _create_generator(self):
        return self.data_reader.load_images()

    def train(self, epochs, batch_size, number_of_images=128, content_loss_weight=1e-3, style_loss_weight=1e-5):

        # Define a Keras callback
        class TestImage(tf.keras.callbacks.Callback):
            def __init__(self, image):
                super(TestImage, self).__init__()
                self.image = image
            
            @staticmethod
            def _show_image(image, title=None):
                if len(image.shape) > 3:
                    image = tf.squeeze(image, axis=0)
                image = tf.image.convert_image_dtype(image, tf.uint8)
                
                plt.imshow(image)
                if title:
                    plt.title(title)

                plt.show()

            def on_epoch_begin(self, epoch, logs=None):
                self._show_image(self.image)

            # Print the accuracy at the end of each epoch
            def on_epoch_end(self, epoch, logs=None):
                output = self.model(self.image)
                image = tf.image.convert_image_dtype(output, tf.uint8)
                image = tf.squeeze(image)
                tf.keras.preprocessing.image.save_img('test_images/epoch' + str(epoch) + '.png',image)
                self._show_image(output)

        train_data_generator = partial(self._create_generator)
        dataset = tf.data.Dataset.from_generator(train_data_generator, output_types = (tf.float32, tf.float32), output_shapes = ((512, 512, 3), (512, 512, 3)))
        dataset = dataset.take(number_of_images).batch(batch_size).prefetch(batch_size//2)
        
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope(): 
        model = ResNet()

        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=StyleContentLoss(self.target_style_image, content_loss_weight, style_loss_weight))
        
        history = model.fit(dataset, epochs = epochs, callbacks = [TestImage(self.test_content_image)])
        print(history.history['loss'])
        # plt.plot(range(epochs), np.log10(history.history['loss']))
        # plt.show()


if __name__=="__main__":
    NST = NeuralStyleTransfer("Kandinsky_Composition_7")
    NST.train(2, 1, 4)