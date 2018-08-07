import unittest
import numpy
from keras_bi_lm import BiLM


class TestInitModel(unittest.TestCase):

    def test_init_single(self):
        bi_lm = BiLM(token_num=101, rnn_layer_num=1, rnn_type='gru')
        bi_lm.model.summary()

    def test_init_multi(self):
        bi_lm = BiLM(token_num=102, rnn_layer_num=3, rnn_keep_num=3)
        bi_lm.model.summary()

    def test_init_multi_keep(self):
        bi_lm = BiLM(token_num=103, rnn_layer_num=6, rnn_keep_num=3)
        bi_lm.model.summary()

    def test_bidirectional(self):
        bi_lm = BiLM(token_num=104, rnn_layer_num=1, use_bidirectional=True)
        bi_lm.model.summary()

    def test_embedding_weights(self):
        bi_lm = BiLM(token_num=105,
                     rnn_layer_num=1,
                     embedding_dim=106,
                     embedding_weights=numpy.random.random((105, 106)))
        bi_lm.model.summary()

    def test_no_embedding(self):
        bi_lm = BiLM(token_num=107,
                     rnn_layer_num=1,
                     embedding_dim=108,
                     has_embedding=False)
        bi_lm.model.summary()
