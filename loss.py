import tensorflow as tf
from model import VGG19

class StyleContentLoss(tf.keras.losses.Loss):
    def __init__(self, target_style_image, content_loss_weight=1e-3, style_loss_weight = 1e-5, name="combined_content_and_style_loss"):
        super().__init__(name=name)
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.vgg19 = VGG19()
        _, target_style_features = self._get_style_and_content_feature_representations(target_style_image)
        self.target_gram_style_features = [self._compute_gram_matrix(style_feature) for style_feature in target_style_features] 

    def _get_style_and_content_feature_representations(self, input_image):
        outputs = self.vgg19(input_image)
        content_features = [content_layer[0] for content_layer in outputs[self.vgg19.num_style_layers:]]
        style_features = [style_layer[0] for style_layer in outputs[:self.vgg19.num_style_layers]]
        return content_features, style_features

    def _compute_content_loss(self, input_image, target):
        return tf.math.reduce_sum(tf.math.square(target-input_image))

    @staticmethod
    def _compute_gram_matrix(input_tensor):
        result = tf.linalg.einsum('ijc,ijd->cd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        # print(tf.shape(result))
        return result/(num_locations)

    def _compute_style_loss(self, style_feature, target_feature):
        return tf.math.reduce_sum(tf.square(style_feature-target_feature))

    def call(self, content_image, model_output_image):

        target_content_features, _ = self._get_style_and_content_feature_representations(content_image)
        content_output_features, style_output_features = self._get_style_and_content_feature_representations(model_output_image)

        style_outputs = [self._compute_gram_matrix(style_output) for style_output in style_output_features]

        style_score, content_score = (0, 0)

        for style_output, gram_target_feature in zip(style_outputs, self.target_gram_style_features):
            style_score += self._compute_style_loss(style_output, gram_target_feature)
        
        for output_content_feature, target_content_feature in zip(content_output_features, target_content_features):
            content_score += self._compute_content_loss(output_content_feature, target_content_feature)

        style_score *= self.style_loss_weight
        content_score *= self.content_loss_weight

        total_score = style_score + content_score

        return total_score