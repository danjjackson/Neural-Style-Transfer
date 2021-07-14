import os
import tensorflow as tf
import numpy as np

from PIL import Image

class Preprocessing:
    def __init__(self, image_directory, out_folder):
        self.image_directory = image_directory
        self.preprocessed_directory = os.path.join(image_directory, out_folder)

    def walk(self, directory, limit = 8192):
        print(directory)
        for root, dirs, files in os.walk(directory):
            for count, file_name in enumerate(files):
                if count > limit:
                    break
                else:
                    path = os.path.join(directory, file_name)
                    self.preprocess_image(path, file_name)

    def preprocess_image(self, image_path, file_name):

        SQUARE_DIM = 512

        image = Image.open(image_path)
        width, height = image.size

        if width > height:
            scale = SQUARE_DIM/height
            new_width = round(width * scale)
            image = image.resize((new_width, SQUARE_DIM))
            image = image.crop(((new_width - SQUARE_DIM)//2, 0, (new_width - SQUARE_DIM)//2 + SQUARE_DIM, 512))

        else:
            scale = SQUARE_DIM/width
            new_height = round(height * scale)
            image = image.resize((SQUARE_DIM, height))
            image = image.crop((0, (new_height - SQUARE_DIM)//2, 512, (new_height - SQUARE_DIM)//2 + SQUARE_DIM))

        image = tf.keras.preprocessing.image.img_to_array(image)
        if np.shape(image)[2] == 3:
            image = tf.keras.applications.vgg19.preprocess_input(image)
            file_name, extension = os.path.splitext(file_name)
            np.save(os.path.join(self.out_directory, file_name), image)

    def store_samples(self):
        """
        Read audio files from `directory` and store the preprocessed version in preprocessed/`directory`

        Args:
        directory: the sub-directory to read from
        preprocess_fnc: The preprocessing function to use

        """
        data_directory = os.path.join(self.image_directory, 'test2014')
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)
            self.walk(data_directory)

        else:
            print('Processed data already exists')

    def load_images(self):
        """
        Load the preprocessed samples from `directory` and return an iterator

        Args:
        directory: the sub-directory to use
        max_size: the maximum audio time length, all others are discarded (default: False)
        loop_infinitely: after one pass, shuffle and pass again (default: False)
        limit_count: maximum number of samples to use, 0 equals unlimited (default: 0)
        feature_type: features to use 'mfcc' or 'power'

        Returns: iterator for samples (audio_data, transcript)

        """

        if not os.path.exists(self.preprocessed_directory):
            raise ValueError('Directory {} does not exist'.format(self.preprocessed_directory))

        for root, dirs, files in os.walk(self.preprocessed_directory):
            for file_name in files:
                file_name = os.path.join(self.preprocessed_directory, file_name)
                image = np.load(file_name)
                yield image/255, image/255


if __name__ == "__main__":
    preprocess = Preprocessing('images', 'preprocessed_images')
    preprocess.store_samples()