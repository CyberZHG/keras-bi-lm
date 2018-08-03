import unittest
from keras_bi_lm import BiLM


class TestInitModel(unittest.TestCase):

    def test_init_single(self):
        bi_lm = BiLM(token_num=101, rnn_layer_num=1, rnn_type='gru')
        bi_lm.model.summary()

    def test_init_multi(self):
        bi_lm = BiLM(token_num=101, rnn_layer_num=3, rnn_keep_num=3)
        bi_lm.model.summary()

    def test_init_multi_keep(self):
        bi_lm = BiLM(token_num=101, rnn_layer_num=6, rnn_keep_num=3)
        bi_lm.model.summary()
