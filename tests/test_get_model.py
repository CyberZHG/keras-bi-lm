import unittest
from keras_bi_lm import get_model


class TestGetModel(unittest.TestCase):

    def test_init_single(self):
        model = get_model(token_num=101, rnn_layer_num=1, rnn_type='gru')
        model.summary()

    def test_init_multi(self):
        model = get_model(token_num=101, rnn_layer_num=3, rnn_keep_num=3)
        model.summary()

    def test_init_multi_keep(self):
        model = get_model(token_num=101, rnn_layer_num=6, rnn_keep_num=3)
        model.summary()
