import unittest
import keras
from keras_bi_lm import BiLM


class TestGetFeatureLayers(unittest.TestCase):

    def test_one_layer(self):
        bi_lm = BiLM(token_num=101, rnn_layer_num=1, rnn_units=50, rnn_type='gru')
        input_layer, output_layer = bi_lm.get_feature_layers()
        self.assertEqual((None, None), input_layer._keras_shape)
        self.assertEqual((None, None, 100), output_layer._keras_shape)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()

    def test_bidirectional(self):
        bi_lm = BiLM(token_num=102, rnn_layer_num=1, rnn_units=50, rnn_type='gru', use_bidirectional=True)
        input_layer, output_layer = bi_lm.get_feature_layers()
        self.assertEqual((None, None), input_layer._keras_shape)
        self.assertEqual((None, None, 200), output_layer._keras_shape)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()

    def test_multiple_layers(self):
        bi_lm = BiLM(token_num=103, rnn_layer_num=6, rnn_keep_num=3, rnn_units=50, rnn_type='lstm')
        input_layer, output_layer = bi_lm.get_feature_layers()
        self.assertEqual((None, None), input_layer._keras_shape)
        self.assertEqual((None, None, 300), output_layer._keras_shape)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()
