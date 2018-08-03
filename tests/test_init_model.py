import unittest
from keras_bi_lm import BiLM


class TestInitModel(unittest.TestCase):

    def test_init_single(self):
        BiLM(token_num=101, rnn_layer_num=1, rnn_type='gru')

    def test_init_multi(self):
        BiLM(token_num=101, rnn_layer_num=3, rnn_keep_num=3)

    def test_init_multi_keep(self):
        BiLM(token_num=101, rnn_layer_num=6, rnn_keep_num=3)
