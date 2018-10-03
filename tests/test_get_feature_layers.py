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
        for layer in bi_lm.model.layers:
            try:
                new_layer = model.get_layer(name=layer.name)
                self.assertEqual(layer.get_weights(), new_layer.get_weights())
            except ValueError:
                pass

    def test_input_layer(self):
        input_layer = keras.layers.Input((None,), name='New-Input')
        bi_lm = BiLM(token_num=104, rnn_layer_num=6, rnn_keep_num=3, rnn_units=50, rnn_type='gru')
        output_layer = bi_lm.get_feature_layers(input_layer=input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()
        for layer in bi_lm.model.layers:
            try:
                new_layer = model.get_layer(name=layer.name)
                self.assertEqual(layer.get_weights(), new_layer.get_weights())
            except ValueError:
                pass

    def test_no_embedding(self):
        input_layer = keras.layers.Input((None, 106), name='New-Input')
        bi_lm = BiLM(token_num=105,
                     has_embedding=False,
                     embedding_dim=106,
                     rnn_layer_num=6,
                     rnn_keep_num=1,
                     rnn_units=50,
                     rnn_type='lstm')
        output_layer = bi_lm.get_feature_layers(input_layer=input_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()
        for layer in bi_lm.model.layers:
            try:
                new_layer = model.get_layer(name=layer.name)
                self.assertEqual(layer.get_weights(), new_layer.get_weights())
            except ValueError:
                pass

    def test_weighted_sum(self):
        bi_lm = BiLM(token_num=107,
                     embedding_dim=108,
                     rnn_layer_num=6,
                     rnn_keep_num=7,
                     rnn_units=108,
                     rnn_type='lstm')
        input_layer, output_layer = bi_lm.get_feature_layers(use_weighted_sum=True)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.summary()
